
// Copyright 2019 Lawrence Livermore National Security. 
// Produced at the Lawrence Livermore National Laboratory.
// LLNL-CODE-781679. All Rights reserved. See file LICENSE for details.
//
// This file is part of graph-embed. For more information and source code
// availability, see github.com/LLNL/graph-embed
//
// SPDX-License-Identifier: LGPL-2.1


#include <algorithm>
#include <random>
#include <tuple>

#include "embed.hpp"
#include "forceatlas.hpp"

#include <omp.h>

namespace partition {

  std::function<void (const SparseMatrix&,
		      const SparseMatrix&,
		      const std::vector<int>&,
		      const std::vector<std::vector<double>>&,
		      const std::vector<double>&,
		      std::vector<std::vector<double>>&,
		      const int)> 
  anyToMultilevel (const std::function<std::vector<std::vector<double>>(const SparseMatrix&, int)> embedder) {
    return [embedder] (const SparseMatrix& A,
		       const SparseMatrix& P_T,
		       const std::vector<int>& v_A,
		       const std::vector<std::vector<double>>& coords_A,
		       const std::vector<double>& r_A,
		       std::vector<std::vector<double>>& coords,
		       const int d) {
      int m = P_T.Rows();
      const std::vector<int>& I = A.GetIndptr();
      const std::vector<int>& J = A.GetIndices();
      const std::vector<int>& P_TI = P_T.GetIndptr();
      const std::vector<int>& P_TJ = P_T.GetIndices();
      for (int a=0; a<m; a++) {
	std::vector<int> v (P_TI[a+1] - P_TI[a]);
	int index = 0;
	for (int c=P_TI[a]; c<P_TI[a+1]; c++) {
	  v[index] = P_TJ[c];
	  index++;
	}
	int r = v.size();
	CooMatrix c (r, r);
	for (int i=0; i<r; i++) {
	  for (int k2=I[v[i]]; k2<I[v[i]+1]; k2++) {
	    int j = J[k2];
	    if (v_A[j] == a) {
	      int jp=-1;
	      for (int j2=0; j2<r; j2++) {
		if (j == v[j2]) {
		  jp = j2;
		}
	      }
	      c.Add(i, jp, 1.0);
	    }
	  }
	}
	std::vector<std::vector<double>> new_coords = embedder(c.ToSparse(), d); 
	// normalize
	double max = 0.0;
	for (int i=0; i<r; i++) {
	  double sum = magnitude(new_coords[i]);
	  if (sum > max) {
	    max = sum;
	  }
	}
	for (int i=0; i<r; i++) {
	  for (int k=0; k<d; k++) {
	    coords[v[i]][k] = coords_A[a][k] + r_A[a] * (new_coords[i][k] / max);
	  }
	}
      }
      return;
    };
  }

  std::vector<std::vector<double>> embedVia(const std::vector<SparseMatrix>& As,
					    const std::vector<SparseMatrix>& P_Ts,
					    const int d,
					    std::function<void (const SparseMatrix&,
								const SparseMatrix&,
								const std::vector<int>&,
								const std::vector<std::vector<double>>&,
								const std::vector<double>&,
								std::vector<std::vector<double>>&,
								const int)> embedder) {

    assert(As.size() == P_Ts.size() + 1);
    for (int i=0; i<P_Ts.size(); i++) {
      assert(As[i].Rows() == P_Ts[i].Cols());
    }
    for (int i=0; i<P_Ts.size(); i++) {
      assert(As[i+1].Rows() == P_Ts[i].Rows());
    }
    std::vector<double> none (0);
    std::vector<std::vector<double>> none2 (0);
    return embedViaMultilevel(As, P_Ts, d, 0, none, none2, embedder);
  }

  std::vector<std::vector<double>> embedViaMultilevel (const std::vector<SparseMatrix>& As,
						       const std::vector<SparseMatrix>& P_Ts,
						       const int d,
						       const int levelIndex,
						       std::vector<double>& r_A,
						       std::vector<std::vector<double>>& coords_A,
					    std::function<void (const SparseMatrix&,
								const SparseMatrix&,
								const std::vector<int>&,
								const std::vector<std::vector<double>>&,
								const std::vector<double>&,
								std::vector<std::vector<double>>&,
								const int)> embedder) {
    if (levelIndex == P_Ts.size()) {
      std::cout << "embedding layer " << levelIndex+1 << ": getting base coords" << std::endl;
      r_A = std::vector<double> (0);
      coords_A = std::vector<std::vector<double>> (0);
      int n = As[levelIndex].Rows();
      std::vector<int> P_TI (n+1);
      for (int i=0; i<n+1; i++) {
	P_TI[i] = i;
      }
      std::vector<int> P_TJ (n, 0);
      std::vector<double> P_TD (n, 1.0);
      SparseMatrix P_T (P_TI, P_TJ, P_TD, 1, n);
      std::vector<std::vector<double>> coords (n);
      std::vector<int> v_A (n, 0);
      std::vector<std::vector<double>> origin = {std::vector<double>(d, 0.0)};
      std::vector<double> r_A = {1.0};
      embedder(As[levelIndex], P_T, v_A, origin, r_A, coords, d);
      return coords;
    } else {

      //========
      std::vector<double> r_Ac (0);
      std::vector<std::vector<double>> coords_Ac (0);
      coords_A = embedMultilevel (As, P_Ts, d, levelIndex+1, r_Ac, coords_Ac);
      //=======

      const double inf = std::numeric_limits<double>::infinity();

      const SparseMatrix& P_T = P_Ts[levelIndex];
      const SparseMatrix& A = As[levelIndex];
      const SparseMatrix& A_c = As[levelIndex+1];
      const std::vector<int>& I = A.GetIndptr();
      const std::vector<int>& J = A.GetIndices();
      const std::vector<int>& P_TI = P_T.GetIndptr();
      const std::vector<int>& P_TJ = P_T.GetIndices();
      const std::vector<int> vertex_A = P_T.Transpose().GetIndices();
      const std::vector<int>& A_cI = A_c.GetIndptr();
      const std::vector<int>& A_cJ = A_c.GetIndices();


      int n = A.Rows();
      int m = coords_A.size();
    
      std::cout << "embeding layer " << levelIndex+1 << std::endl;

      r_A = std::vector<double> (m, 0.0);
      if (r_Ac.size() == 0) {
	// Base case: no coordinates computed yet.
	std::vector<std::tuple<double, int, int>> times;
	for (int i=0; i<m; i++) {
	  bool doAll = true;
	  if (doAll) {
	    for (int j=i+1; j<m; j++) {
	      double distance_ij = distance(coords_A[i], coords_A[j]); 
	      times.push_back(std::tuple<double, int, int>(-distance_ij/2, i, j));
	    }
	  } else {
	    for (int kk=A_cI[i]; kk<A_cI[i+1]; kk++) {
	      int j = A_cJ[kk];
	      if (i < j) {
		double distance_ij = distance(coords_A[i], coords_A[j]);
		times.push_back(std::tuple<double, int, int> (-distance_ij/2, i, j));
	      }
	    }
	  }
	}
	std::sort(times.begin(), times.end());
	int count = 0; // how many coordinates have been assigned
	while (count < m && times.size() != 0) {
	  int i, j, i_prime, j_prime;
	  double time_ij;
	  std::tie(time_ij, i, j) = times.back();
	  double distance = -time_ij;
	  times.pop_back();
	
	  if (r_A[i] <= 0.0 && r_A[j] > 0.0) { // only i is live
	    r_A[i] = distance;
	    for (int a=0; i>=0 && a<times.size(); a++) {
	      std::tie(std::ignore, i_prime, j_prime) = times[a];
	      if (i_prime == i || j_prime == i) {
		std::get<0>(times[a]) = - (2 * (-std::get<0>(times[a])) - (-time_ij));
	      }
	    }
	    std::sort(times.begin(), times.end());
	    count++;
	  } else if (r_A[i] > 0.0 && r_A[j] <= 0.0) { // only j is live
	    r_A[j] = distance;
	    for (int a=0; j>=0 && a<times.size(); a++) {
	      std::tie(std::ignore, i_prime, j_prime) = times[a];
	      if (i_prime == j || j_prime == j) {
		std::get<0>(times[a]) = - (2 * (-std::get<0>(times[a])) - (-time_ij));
	      }
	    }
	    std::sort(times.begin(), times.end());
	    count++;
	  } else if (r_A[i] <= 0 && r_A[j] <= 0) { // both are live
	    r_A[i] = distance;
	    r_A[j] = distance;
	    for (int a=0; a<times.size(); a++) {
	      std::tie(std::ignore, i_prime, j_prime) = times[a];
	      if (i_prime == i || j_prime == i || i_prime == j || j_prime == j) {
		std::get<0>(times[a]) = - (2 * (-std::get<0>(times[a])) - (-time_ij));
	      }
	    }
	    std::sort(times.begin(), times.end());
	    count += 2;
	  } else { // both are dead
	  }
	}
      
      } else {
	const std::vector<int>& P_TI_c = P_Ts[levelIndex+1].GetIndptr();
	const std::vector<int>& P_TJ_c = P_Ts[levelIndex+1].GetIndices();
	int mc = P_Ts[levelIndex+1].Rows();
	std::vector<int> vertex_Ac = P_Ts[levelIndex+1].Transpose().GetIndices();
	#pragma omp parallel for
	for (int b=0; b<mc; b++) {
	  std::vector<std::tuple<double, int, int>> times;
	  int s = P_TI_c[b+1] - P_TI_c[b];
	  bool doAll = false;
	  for (int i=0; i<s; i++) {
	    if (doAll) {
	      for (int j=i+1; j<s; j++) {
		double distance_ij = distance(coords_A[P_TJ_c[P_TI_c[b] + i]], coords_A[P_TJ_c[P_TI_c[b] + j]]); 
		times.push_back(std::tuple<double, int, int>(-distance_ij/2, P_TJ_c[P_TI_c[b] + i], P_TJ_c[P_TI_c[b] + j]));
	      }
	    } else {
	      int a = P_TJ_c[P_TI_c[b] + i];
	      for (int kk=A_cI[a]; kk<A_cI[a+1]; kk++) {
	      	int j = A_cJ[kk];
	      	if (a < j && vertex_Ac[j] == vertex_Ac[a]) {
	        double distance_ij = distance(coords_A[a], coords_A[j]);
	      	  times.push_back(std::tuple<double, int, int> (-distance_ij/2, a, j));
	      	}
	      }
	    }
	  }
	  if (s == 1) {
	    assert(times.size() == 0);
	    r_A[P_TJ_c[P_TI_c[b]]] = r_Ac[b];
	    continue;
	  }
	  
	  std::sort(times.begin(), times.end());
	  int count = 0; // how many coordinates have been assigned
	  while (count < m && times.size() != 0) {
	    int i, j, i_prime, j_prime;
	    double time_ij;
	    std::tie(time_ij, i, j) = times.back();
	    double distance = -time_ij;
	    times.pop_back();
	
	    if (r_A[i] <= 0.0 && r_A[j] > 0.0) { // only i is live
	      r_A[i] = distance;
	      for (int r=0; i>=0 && r<times.size(); r++) {
		std::tie(std::ignore, i_prime, j_prime) = times[r];
		if (i_prime == i || j_prime == i) {
		  std::get<0>(times[r]) = - (2 * (-std::get<0>(times[r])) - (-time_ij));
		}
	      }
	      std::sort(times.begin(), times.end());
	      count++;
	    } else if (r_A[i] > 0.0 && r_A[j] <= 0.0) { // only j is live
	      r_A[j] = distance;
	      for (int r=0; j>=0 && r<times.size(); r++) {
		std::tie(std::ignore, i_prime, j_prime) = times[r];
		if (i_prime == j || j_prime == j) {
		  std::get<0>(times[r]) = - (2 * (-std::get<0>(times[r])) - (-time_ij));
		}
	      }
	      std::sort(times.begin(), times.end());
	      count++;
	    } else if (r_A[i] <= 0 && r_A[j] <= 0) { // both are live
	      r_A[i] = distance;
	      r_A[j] = distance;
	      for (int r=0; r<times.size(); r++) {
		std::tie(std::ignore, i_prime, j_prime) = times[r];
		if (i_prime == i || j_prime == i || i_prime == j || j_prime == j) {
		  std::get<0>(times[r]) = - (2 * (-std::get<0>(times[r])) - (-time_ij));
		}
	      }
	      std::sort(times.begin(), times.end());
	      count += 2;
	    } else { // both are dead
	    }
	  }
	}
	for (int b=0; b<mc; b++) {
	  double alpha = 0.0;
	  for (int k2=P_TI_c[b]; k2<P_TI_c[b+1]; k2++) {
	    int a = P_TJ_c[k2];
	    double dis = distance(coords_Ac[b], coords_A[a]) + r_A[a];
	    if (dis > alpha) {
	      alpha = dis;
	    }
	  }
	  double epsilon = 0.000001;
	  if (alpha < epsilon) {
	    alpha = epsilon;
	  }
	  for (int k2=P_TI_c[b]; k2<P_TI_c[b+1]; k2++) {
	    int a = P_TJ_c[k2];
	    for (int k=0; k<d; k++) {
	      coords_A[a][k] = coords_Ac[b][k] + (r_Ac[b] / alpha) * (coords_A[a][k] - coords_Ac[b][k]);
	    }
	    r_A[a] = (r_Ac[b] / alpha) * r_A[a];
	  }
	}
      }

      std::vector<std::vector<double>> coords (n, std::vector<double> (d));
      embedder(A, P_T, vertex_A, coords_A, r_A, coords, d);
      return coords;
    }
  }

  // =================
  // -----------------
  // =================
  
  std::vector<std::vector<double>> embedViaMinimization (const SparseMatrix& A, const int d) {
    std::vector<std::vector<double>> coords (A.Rows(), std::vector<double>(d, 0.0));
    embedViaMinimization(A, d, coords, 1000);
    return coords;
  }

  void embedViaMinimization (const SparseMatrix& A, const int d, std::vector<std::vector<double>>& new_coords, const int ITER) {
    double inf = std::numeric_limits<double>::infinity();
    double epsilon = 10e-12;

    std::random_device rd;
    std::mt19937 gen (rd());
    //unsigned seed = 0;
    //fstd::default_random_engine gen (seed);
    std::uniform_real_distribution<double> random (-1.0, 1.0);
    
    int n = A.Rows();
    
    const std::vector<int>& I = A.GetIndptr();
    const std::vector<int>& J = A.GetIndices();
    const std::vector<double>& D = A.GetData();
  
    if (new_coords.size() == 0) {
      new_coords = std::vector<std::vector<double>> (n, std::vector<double> (d));
      // initialize x_i
      for (int i=0; i<n; i++) {
	for (int k=0; k<d; k++) {
	  new_coords[i][k] = random(gen);
	}
      }
    }
    
    std::vector<std::vector<double>> dirs;
    if (d == 2) {
      dirs = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
      //dirs = {{0, 1.0}, {-sqrt(3.0)/2.0, -0.5}, {sqrt(3.0)/2.0, -0.5}};
    } else if (d == 3) {
      dirs = {{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}};
    } else {
      for (int k=0; k<d; k++) {
	std::vector<double> e_k (d, 0.0);
	e_k[k] = 1;
	dirs.push_back(e_k);
	e_k[k] = -1;
	dirs.push_back(e_k);
      }
    }

    //for (int i=0; i<dirs.size(); i++) {
    //  for (int k=0; k<d; k++) {
    //	dirs[i][k] *= n;
    //  }
    //}

    // iterate to find solution
    for (int iter=0; iter<ITER; iter++) {
      for (int i=0; i<n; i++) {
	std::vector<double>& x_i = new_coords[i];

	int count = 0;
	for (int kk=I[i]; kk<I[i+1]; kk++) {
	  if (J[kk] != i) {
	    count++;
	  }
	}
	if (count != 0) {
	  double min_J_loc = inf;
	  double min_t = 0.0f;
	  double min_s = -1;
	  double w = 1000000.0;
	  for (int s=0; s<dirs.size(); s++) {
	    // find the minimum for the interval (x_i, x_s)
	    const std::vector<double>& x_s = dirs[s];

	    double t = 0.5;
	    double jump = 0.25;
	  
	    do {
	      double dJ_loc_dt = 0.0;
	      for (int r=0; r<n; r++) {
		if (i != r) {
		  const auto& x_r = new_coords[r];
		  double term1 = 0.0;
		  double term2 = 0.0;
		  for (int k=0; k<d; k++) {
		    double u_k = x_s[k] - x_i[k];
		    double v_k = x_i[k] - x_r[k];
		    double z_k = (u_k * t + v_k);
		    term1 = term1 + z_k * z_k;
		    term2 = term2 + z_k * u_k;
		  }
		  if (term1 < epsilon) {
		    term1 = epsilon;
		  }
		  dJ_loc_dt += - ((1.0 / sqrt(term1 * term1 * term1)) * term2);
		}
	      }

	      for (int kk=I[i]; kk<I[i+1]; kk++) {
		int r = J[kk];
		if (i != r) {
		  const auto& x_r = new_coords[r];
		  double term = 0.0;
		  for (int k=0; k<d; k++) {
		    double a = (1 - t) * x_i[k] + t * x_s[k] - x_r[k];
		    term += w * 2.0 * a * (x_s[k] - x_i[k]);
		  }
		  dJ_loc_dt += term;
		}
	      }
	  
	      if (dJ_loc_dt < 0.0) {
		t = t + jump;
	      } else {
		t = t - jump;
	      }
	      jump = jump / 2.0;
	    
	    } while (jump > 1.e-4);
	  
	    double J_loc = 0.0;
	    for (int r=0; r<n; r++) {
	      if (i != r) {
		const auto& x_r = new_coords[r];
		double term1 = 0.0;
		for (int k=0; k<d; k++) {
		  double u_k = x_s[k] - x_i[k];
		  double v_k = x_i[k] - x_r[k];
		  double z_k = (u_k * t + v_k);
		  term1 = term1 + z_k * z_k;
		}
		if (term1 < epsilon) {
		  term1 = epsilon;
		}
		J_loc = J_loc + 1.0 / sqrt(term1);
	      }
	    }
	    for (int kk=I[i]; kk<I[i+1]; kk++) {
	      int r = J[kk];
	      if (i != r) {
		const auto& x_r = new_coords[r];
		double term = 0.0;
		for (int k=0; k<d; k++) {
		  double a = (1 - t) * x_i[k] + t * x_s[k] - x_r[k];
		  term += a * a;
		}
		J_loc += w * term;
	      }
	    }
	    if (J_loc < min_J_loc) {
	      min_J_loc = J_loc;
	      min_t = t;
	      min_s = s;
	    }

	    //int num = 100;
	    //double dt = 1.0 / num;
	    //for (int a=1; a<num; a++) {
	    //  double t = a * dt;
	    //  double J_loc = 0.0;
	    //  for (int r=0; r<s_num; r++) {
	    //    auto& x_r = x_j[r];
	    //    double bottom = 0.0;
	    //    for (int k=0; k<d; k++) {
	    //	double z_k = (1 - t) * x_i[k] + t * x_s[k] - x_r[k];
	    //    	bottom = bottom + z_k * z_k;
	    //    }
	    //    J_loc = J_loc + 1.0 / sqrt(bottom);
	    //  }
	    //  if (J_loc > min_J_loc) {
	    //    min_J_loc = J_loc;
	    //    min_t = t;
	    //    min_s = s;
	    //  }
	    //}
	    
	  }
	  //std::cout << "i: " << i << " | t: " << min_t << " | s: " << min_s << std::endl;
	  if (min_s >= 0) {
	    for (int k=0; k<d; k++) {
	      new_coords[i][k] = new_coords[i][k] * (1 - min_t) + dirs[min_s][k] * min_t;
	    }
	  }
	}
      }
    }

    if (n > 1) {
      // center coords at 0
      std::vector<double> avg (d);
      for (int i=1; i<n; i++) {
	for (int k=0; k<d; k++) {
	  avg[k] = avg[k] + new_coords[i][k];
	}
      }
      for (int k=0; k<d; k++) {
	avg[k] = avg[k] / (n - 1);
      }
      for (int i=0; i<n; i++) {
	for (int k=0; k<d; k++) {
	  new_coords[i][k] -= avg[k];
	}
      }

      double max_length = 0.0;
      for (int i=1; i<n; i++) {
	double length = magnitude(new_coords[i]);
	if (max_length < length) {
	  max_length = length;
	}
      }
      for (int i=0; i<n; i++) {
	for (int k=0; k<d; k++) {
	  new_coords[i][k] = new_coords[i][k] / max_length;
	}
      }
    }
    return;
  }

  std::vector<std::vector<double>> embed (const std::vector<SparseMatrix>& As,
					  const std::vector<SparseMatrix>& P_Ts,
					  const int d) {
    assert(As.size() == P_Ts.size() + 1);
    for (int i=0; i<P_Ts.size(); i++) {
      assert(As[i].Rows() == P_Ts[i].Cols());
    }
    for (int i=0; i<P_Ts.size(); i++) {
      assert(As[i+1].Rows() == P_Ts[i].Rows());
    }
    std::vector<double> none (0);
    std::vector<std::vector<double>> none2 (0);
    return embedMultilevel(As, P_Ts, d, 0, none, none2);
  }    

  std::vector<std::vector<double>> embedMultilevel (const std::vector<SparseMatrix>& As,
						    const std::vector<SparseMatrix>& P_Ts,
						    const int d,
						    const int levelIndex,
						    std::vector<double>& r_A,
						    std::vector<std::vector<double>>& coords_A) {
    if (levelIndex == P_Ts.size()) {
      std::cout << "embedding layer " << levelIndex+1 << ": getting base coords" << std::endl;
      r_A = std::vector<double> (0);
      coords_A = std::vector<std::vector<double>> (0);
      std::vector<std::vector<double>> coords = partition::forceAtlas(As[levelIndex], d);
      return coords;
    } else {

      //========
      std::vector<double> r_Ac (0);
      std::vector<std::vector<double>> coords_Ac (0);
      coords_A = embedMultilevel (As, P_Ts, d, levelIndex+1, r_Ac, coords_Ac);
      //=======

      const double inf = std::numeric_limits<double>::infinity();

      const SparseMatrix& P_T = P_Ts[levelIndex];
      const SparseMatrix& A = As[levelIndex];
      const SparseMatrix& A_c = As[levelIndex+1];
      const std::vector<int>& I = A.GetIndptr();
      const std::vector<int>& J = A.GetIndices();
      const std::vector<int>& P_TI = P_T.GetIndptr();
      const std::vector<int>& P_TJ = P_T.GetIndices();
      const std::vector<int> vertex_A = P_T.Transpose().GetIndices();
      const std::vector<int>& A_cI = A_c.GetIndptr();
      const std::vector<int>& A_cJ = A_c.GetIndices();


      int n = A.Rows();
      int m = coords_A.size();
    
      std::cout << "embeding layer " << levelIndex+1 << std::endl;

      r_A = std::vector<double> (m, 0.0);
      if (r_Ac.size() == 0) {
	// Base case: no coordinates computed yet.
	std::vector<std::tuple<double, int, int>> times;
	for (int i=0; i<m; i++) {
	  bool doAll = true;
	  if (doAll) {
	    for (int j=i+1; j<m; j++) {
	      double distance_ij = distance(coords_A[i], coords_A[j]); 
	      times.push_back(std::tuple<double, int, int>(-distance_ij/2, i, j));
	    }
	  } else {
	    for (int kk=A_cI[i]; kk<A_cI[i+1]; kk++) {
	      int j = A_cJ[kk];
	      if (i < j) {
		double distance_ij = distance(coords_A[i], coords_A[j]);
		times.push_back(std::tuple<double, int, int> (-distance_ij/2, i, j));
	      }
	    }
	  }
	}
	std::sort(times.begin(), times.end());
	int count = 0; // how many coordinates have been assigned
	while (count < m && times.size() != 0) {
	  int i, j, i_prime, j_prime;
	  double time_ij;
	  std::tie(time_ij, i, j) = times.back();
	  double distance = -time_ij;
	  times.pop_back();
	
	  if (r_A[i] <= 0.0 && r_A[j] > 0.0) { // only i is live
	    r_A[i] = distance;
	    for (int a=0; i>=0 && a<times.size(); a++) {
	      std::tie(std::ignore, i_prime, j_prime) = times[a];
	      if (i_prime == i || j_prime == i) {
		std::get<0>(times[a]) = - (2 * (-std::get<0>(times[a])) - (-time_ij));
	      }
	    }
	    std::sort(times.begin(), times.end());
	    count++;
	  } else if (r_A[i] > 0.0 && r_A[j] <= 0.0) { // only j is live
	    r_A[j] = distance;
	    for (int a=0; j>=0 && a<times.size(); a++) {
	      std::tie(std::ignore, i_prime, j_prime) = times[a];
	      if (i_prime == j || j_prime == j) {
		std::get<0>(times[a]) = - (2 * (-std::get<0>(times[a])) - (-time_ij));
	      }
	    }
	    std::sort(times.begin(), times.end());
	    count++;
	  } else if (r_A[i] <= 0 && r_A[j] <= 0) { // both are live
	    r_A[i] = distance;
	    r_A[j] = distance;
	    for (int a=0; a<times.size(); a++) {
	      std::tie(std::ignore, i_prime, j_prime) = times[a];
	      if (i_prime == i || j_prime == i || i_prime == j || j_prime == j) {
		std::get<0>(times[a]) = - (2 * (-std::get<0>(times[a])) - (-time_ij));
	      }
	    }
	    std::sort(times.begin(), times.end());
	    count += 2;
	  } else { // both are dead
	  }
	}
      
      } else {
	const std::vector<int>& P_TI_c = P_Ts[levelIndex+1].GetIndptr();
	const std::vector<int>& P_TJ_c = P_Ts[levelIndex+1].GetIndices();
	int mc = P_Ts[levelIndex+1].Rows();
	std::vector<int> vertex_Ac = P_Ts[levelIndex+1].Transpose().GetIndices();
	#pragma omp parallel for
	for (int b=0; b<mc; b++) {
	  std::vector<std::tuple<double, int, int>> times;
	  int s = P_TI_c[b+1] - P_TI_c[b];
	  bool doAll = false;
	  for (int i=0; i<s; i++) {
	    if (doAll) {
	      for (int j=i+1; j<s; j++) {
		double distance_ij = distance(coords_A[P_TJ_c[P_TI_c[b] + i]], coords_A[P_TJ_c[P_TI_c[b] + j]]); 
		times.push_back(std::tuple<double, int, int>(-distance_ij/2, P_TJ_c[P_TI_c[b] + i], P_TJ_c[P_TI_c[b] + j]));
	      }
	    } else {
	      int a = P_TJ_c[P_TI_c[b] + i];
	      for (int kk=A_cI[a]; kk<A_cI[a+1]; kk++) {
	      	int j = A_cJ[kk];
	      	if (a < j && vertex_Ac[j] == vertex_Ac[a]) {
	        double distance_ij = distance(coords_A[a], coords_A[j]);
	      	  times.push_back(std::tuple<double, int, int> (-distance_ij/2, a, j));
	      	}
	      }
	    }
	  }
	  if (s == 1) {
	    assert(times.size() == 0);
	    r_A[P_TJ_c[P_TI_c[b]]] = r_Ac[b];
	    continue;
	  }
	  
	  std::sort(times.begin(), times.end());
	  int count = 0; // how many coordinates have been assigned
	  while (count < m && times.size() != 0) {
	    int i, j, i_prime, j_prime;
	    double time_ij;
	    std::tie(time_ij, i, j) = times.back();
	    double distance = -time_ij;
	    times.pop_back();
	
	    if (r_A[i] <= 0.0 && r_A[j] > 0.0) { // only i is live
	      r_A[i] = distance;
	      for (int r=0; i>=0 && r<times.size(); r++) {
		std::tie(std::ignore, i_prime, j_prime) = times[r];
		if (i_prime == i || j_prime == i) {
		  std::get<0>(times[r]) = - (2 * (-std::get<0>(times[r])) - (-time_ij));
		}
	      }
	      std::sort(times.begin(), times.end());
	      count++;
	    } else if (r_A[i] > 0.0 && r_A[j] <= 0.0) { // only j is live
	      r_A[j] = distance;
	      for (int r=0; j>=0 && r<times.size(); r++) {
		std::tie(std::ignore, i_prime, j_prime) = times[r];
		if (i_prime == j || j_prime == j) {
		  std::get<0>(times[r]) = - (2 * (-std::get<0>(times[r])) - (-time_ij));
		}
	      }
	      std::sort(times.begin(), times.end());
	      count++;
	    } else if (r_A[i] <= 0 && r_A[j] <= 0) { // both are live
	      r_A[i] = distance;
	      r_A[j] = distance;
	      for (int r=0; r<times.size(); r++) {
		std::tie(std::ignore, i_prime, j_prime) = times[r];
		if (i_prime == i || j_prime == i || i_prime == j || j_prime == j) {
		  std::get<0>(times[r]) = - (2 * (-std::get<0>(times[r])) - (-time_ij));
		}
	      }
	      std::sort(times.begin(), times.end());
	      count += 2;
	    } else { // both are dead
	    }
	  }
	}
	for (int b=0; b<mc; b++) {
	  double alpha = 0.0;
	  for (int k2=P_TI_c[b]; k2<P_TI_c[b+1]; k2++) {
	    int a = P_TJ_c[k2];
	    double dis = distance(coords_Ac[b], coords_A[a]) + r_A[a];
	    if (dis > alpha) {
	      alpha = dis;
	    }
	  }
	  double epsilon = 0.000001;
	  if (alpha < epsilon) {
	    alpha = epsilon;
	  }
	  for (int k2=P_TI_c[b]; k2<P_TI_c[b+1]; k2++) {
	    int a = P_TJ_c[k2];
	    for (int k=0; k<d; k++) {
	      coords_A[a][k] = coords_Ac[b][k] + (r_Ac[b] / alpha) * (coords_A[a][k] - coords_Ac[b][k]);
	    }
	    r_A[a] = (r_Ac[b] / alpha) * r_A[a];
	  }
	}
      }

      std::random_device rd;
      std::mt19937 gen (rd());
      //unsigned seed = 0;
      //std::default_random_engine gen (seed);
      std::uniform_real_distribution<double> random (-1.0, 1.0);

      std::vector<std::vector<double>> coords (n, std::vector<double> (d));
      //for (int i=0; i<n; i++) {
      //  for (int k=0; k<d; k++) {
      //    int a = vertex_A[i];
      //    new_coords[i][k] = coords[a][k] + r_A[a] * random(gen);
      //  }
      //}
      forceAtlasMultilevel(A, P_T, vertex_A, coords_A, r_A, coords, d, 100);
      return coords;
    }  
  } 
}





