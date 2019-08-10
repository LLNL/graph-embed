
// Copyright 2019 Lawrence Livermore National Security. 
// Produced at the Lawrence Livermore National Laboratory.
// LLNL-CODE-781679. All Rights reserved. See file LICENSE for details.
//
// This file is part of graph-embed. For more information and source code
// availability, see github.com/LLNL/graph-embed
//
// SPDX-License-Identifier: LGPL-2.1


#include <cstdlib>
#include <cmath>
//#include <tuple>
#include <limits>
#include <utility>

#include <map>

#include "partitioner.hpp"
#include "matrixutils.hpp"

#include <omp.h>


namespace partition {


  SparseMatrix interpolationMatrix (const int numCols, const std::vector<std::vector<int>>& partition) {
    int numRows = partition.size();
    /* the easy version? (less efficient):
       v  CooMatrix P (numRows, numCols);
  
       int row = 0;
       for (std::vector<int> set : partition) { 
       for (int col : set) {
       std::cout << row << " " << col << std::endl;
       P.Add(row, col, 1.0);
       }
       row++;
       }

       return P.ToSparse();
    */

    std::vector<int> I (numRows+1);
    std::vector<int> J (numCols);
    std::vector<double> D (numCols, 1.0);
      
    int sum = 0;
    I[0] = sum;
    int i = 0;
    int count = 0;
    for (const std::vector<int>& set : partition) {
      for (int j : set) {
	J[count] = j;
	count++;
      }
      sum += set.size();
      I[i+1] = sum;
      i++;
    }
    assert(numCols == count);
    return SparseMatrix (I, J, D, numRows, numCols);
  }


  
  double modularity (const SparseMatrix& A, const SparseMatrix& P_T) {
    auto P = P_T.Transpose();
    //auto L_prime = P.Mult(MatrixUtils::fromLaplacian(L)).Mult(P_T);

    double T = 0.0;
  
    int N = A.Rows();
    int M = P_T.Rows();
  
    std::vector<double> d (M, 0.0);
    std::vector<double> out (M, 0.0);
  
    const std::vector<int>& I = A.GetIndptr();
    const std::vector<int>& J = A.GetIndices();
    const std::vector<double>& D = A.GetData();
  
    const std::vector<int>& agg = P.GetIndices();

    for (int i=0; i<N; i++) {
      for (int k=I[i]; k<I[i+1]; k++) {
	int j = J[k];
	int a_ij = D[k];
	int A = agg[i];
	int B = agg[j];
	
	if (A == B) {
	  d[A] += a_ij;
	  
	} else {
	  out[A] += a_ij;
	}
	
	T += a_ij;
      }
    }
    
    double sum = 0.0;
    for (int A=0; A<M; A++) {
      double alpha_A = (d[A] + out[A]) / T;
      
      sum += d[A]/T - alpha_A * alpha_A;
    }
    double Q = sum;

    return Q;
  }

  SparseMatrix partitionTest (const SparseMatrix& A, const float stallStopThreshold) {

    // at any iteration, if we created a partition it would have M rows and N columns
    int N = A.Rows(); 
    int M = N;
  
    std::vector<float> alpha (N);
    
    const std::vector<int>& I = A.GetIndptr();
    const std::vector<int>& J = A.GetIndices();
    const std::vector<double>& D = A.GetData();
  
    // store edge weights in a vector of trees
    std::vector<std::map<int, float> > a (N);

    #pragma omp parallel for
    for (int i=0; i<N; i++) {
      float alpha_i = 0.0;
      for (int k=I[i]; k<I[i+1]; k++) {
	int j = J[k];
	float a_ij = D[k];
	if (i != j) {
	  a[i].insert(std::make_pair (j, a_ij));
	}

	alpha_i += a_ij;
 
      }
      alpha[i] = alpha_i;
    }

    float d_sum = 0.0;
    float T = 0.0;
    for (int i=0; i<N; i++) {
      for (int k=I[i]; k<I[i+1]; k++) {
	int j = J[k];
	float a_ij = D[k];
	if (i == j) {
	  d_sum += a_ij;
	} 
	T += a_ij;
      }
    }

    #pragma omp parallel for
    for (int i=0; i<alpha.size(); i++) {
      alpha[i] /= T;
    }


    float inf = std::numeric_limits<float>::infinity();
  
    float sum = d_sum / T;
    for (int i=0; i<N; i++) {
      sum += - alpha[i] * alpha[i];
    }
    // initial Q
    // \sum_{i} d_i / T - \alpha_i^2
    float Q = sum; 
    
    // this vector tells us what vertices still exist in the graph after we create a partition
    // for creating a single partition, this vector won't be touched, so be don't need to create it
    // std::vector<int> basis (N);


    // used is a list of the vertices that are still in use
    std::vector<int> used (N);
    // pointer[i] is the index of i in used
    std::vector<int> pointer (N);

    // this is the disjoint sets data structure
    std::vector<int> id (N);


    #pragma omp parallel for
    for (int i=0; i<N; i++) {
      //basis[i] = i;
      used[i] = i;
      pointer[i] = i;
      id[i] = i;
    }

    // stores the maximum \eta (\Delta \Q) value
    std::vector<float> max_DeltaQ (N, 0.0);
    // stores the best index to merge with
    std::vector<float> max_ind (N, -1);

    std::vector<int> mPointer (N);
    std::vector<int> mUsed (N);
    #pragma omp parallel for
    for (int i=0; i<N; i++) {
      mUsed[i] = i;
      mPointer[i] = i;
    }
    int mUsedSize = N;

    M = used.size();
    int M_prev = M;

    linalgcpp::Timer timer (linalgcpp::Timer::Start::True);
    do {    
      assert(M == used.size());
      //assert(N == basis.size());

      // keeps track of the merged pairs    
      std::vector<std::pair<int, int>> merged;

      #pragma omp parallel for
      for (int x=0; x<mUsedSize; x++) {
	int i = mUsed[x];
	if (max_ind[i] == -1) {
	  float max_DeltaQ_i = 0.0;
	  int max_coord = -1;

	  // find the best one to merge with that hasn't been merged
	  for (const auto& kv : a[i]) {
	    int j = kv.first;
	    float a_ij = kv.second;
	    assert(i != j);
	    float DeltaQ_ij = 2 * (a_ij / T - alpha[i] * alpha[j]); 
	    if (DeltaQ_ij > max_DeltaQ_i) {
	      max_DeltaQ_i = DeltaQ_ij;
	      max_coord = j;
	    }
	  }
	  
	  max_DeltaQ[i] = max_DeltaQ_i;
	  max_ind[i] = max_coord;
	}
      }

      float DeltaQ = 0.0;
      for (int x=0; x<mUsedSize; x++) {
	int i = mUsed[x];
	int j = max_ind[i];
	// If the score of the one to merge with is no better than this nodes score, then merge
	if (j != -1 && max_ind[j] == i) {
	  float DeltaQ_ij = max_DeltaQ[i];
	    
	  if (DeltaQ_ij > 0) {
	    if (i < j || !(mPointer[j] < mUsedSize)) {
	      int i_prime, j_prime;
	      if (a[i].size() < a[j].size()) {
		i_prime = j;
		j_prime = i;
	      } else {
		i_prime = i;
		j_prime = j;
	      }
	      
	      merged.push_back(std::make_pair(i_prime,j_prime));
	      
	      DeltaQ += DeltaQ_ij;
	    }
	  }
	}
      }
      
      //{
      //	std::vector<int> test(N, 0);
      //	for (int x=0; x<merged.size(); x++) {
      //	  int i_prime = std::get<0>(merged[x]);
      //	  int j_prime = std::get<1>(merged[x]);
      //	  for (const auto& elt : a[j_prime]) { 
      //	    int k = elt.first; 
      //	    test[k] = 1;
      //	  }
      //	  for (const auto& elt : a[i_prime]) { 
      //	    int k = elt.first; 
      //	    test[k] = 1;
      //	  }
      //	}
      //	int sum = 0;
      //	for (int i=0; i<test.size(); i++) {
      //	  sum += test[i];
      //	}
      //	std::cout << "fraction touched: " << (1.0 * sum) / test.size() << std::endl;
      //}

      // the "matrix multiplication" step, merging eges
      for (int x=0; x<merged.size(); x++) {
	int i_prime = std::get<0>(merged[x]);
	int j_prime = std::get<1>(merged[x]);
	// i_prime is the new representative, j_prime is being removed
	// for every edge connecting j_prime and k, we have add or update the edge connecting i_prime and k
	for (const auto& elt : a[j_prime]) { 
	  int k = elt.first; 
	  float a_jk = elt.second;
	  a[k].erase(a[k].find(j_prime));

	  if (k == i_prime) {
	    //float a_ij = a_jk;
	    //d[i_prime] = d[i_prime] + d[j_prime] + 2*a_ij;
	    alpha[i_prime] = alpha[i_prime] + alpha[j_prime];
	  } else {
	    a[i_prime][k] += a_jk;
	    a[k][i_prime] += a_jk;
	  
	  }
	}
      }

      // ===========

      mUsedSize = 0;
      for (int x=0; x<merged.size(); x++) {
	int i_prime = std::get<0>(merged[x]);
	max_ind[i_prime] = -1;
	if (!(mPointer[i_prime] < mUsedSize)) {
	  int index = mPointer[i_prime];
	  int kp = mUsed[mUsedSize];
	  mUsed[index] = kp;
	  mPointer[kp] = index;
	  mUsed[mUsedSize] = i_prime;
	  mPointer[i_prime] = mUsedSize;
	  mUsedSize++;
	}
	
	for (const auto& elt : a[i_prime]) { 
	  int k = elt.first; 
	  max_ind[k] = -1;

	  if (!(mPointer[k] < mUsedSize)) {
	    int index = mPointer[k];
	    int kp = mUsed[mUsedSize];
	    mUsed[index] = kp;
	    mPointer[kp] = index;
	    mUsed[mUsedSize] = k;
	    mPointer[k] = mUsedSize;
	    mUsedSize++;
	  }
	}
      }

      Q += DeltaQ;

      M_prev = M;

      // remove all j_prime from used, and update the pointers
      for (int x=0; x<merged.size(); x++) {
	int i_prime = std::get<0>(merged[x]);
	int j_prime = std::get<1>(merged[x]);
	int index_j_prime = pointer[j_prime];
	int k = used[used.size() -1];
	std::iter_swap(used.begin() + index_j_prime, used.end() -1);
	used.pop_back();
	pointer[k] = index_j_prime;

	id[j_prime] = i_prime;
    
      }

      M -= merged.size();
      std::cout << "N: " << M << " | " << (1.0 * M) / M_prev << " " << Q << " " << (1.0 * mUsedSize / M) << std::endl;

      //{
      //	std::vector<int> test(N, 0);
      //	for (int x=0; x<merged.size(); x++) {
      //	  int i_prime = std::get<0>(merged[x]);
      //	  test[i_prime] = 1;
      //	  for (const auto& elt : a[i_prime]) { 
      //	    int k = elt.first; 
      //	    test[k] = 1;
      //	  }
      //	}
      //	int sum = 0;
      //	for (int i=0; i<test.size(); i++) {
      //	  sum += test[i];
      //	}
      //	std::cout << "fraction touched: " << (1.0 * sum) / used.size() << std::endl;
      //}

      //for (int x=0; x<mUsedSize; x++) {
      //	assert(pointer[mUsed[x]] < used.size());
      //	std::cout << mUsed[x] << " ";
      //}
      //std::cout << std::endl;
      //for (int i=0; i<N; i++) {
      //	std::cout << i << ": " << mPointer[i] << " " << mUsed[mPointer[i]] << std::endl;
      //}
      //for (int i=0; i<N; i++) {
      //	std::cout << i << ": " << pointer[i] << " " << used.size() << std::endl;
      //}

	//std::cout << std::setprecision(2) << "ratio: " << (1.0 * M)/M_prev << "   \t Q: " << Q << "   \t |{A}|: " << M << std::endl; //<< "   \t avg |a_i|: " << (1.0 * sum) / used.size() << std::endl;

    } // continue until the coarsening factor is large enough; by default only halt if M == M_prev
    while (1.0*M/M_prev < stallStopThreshold);

    timer.Click();
    std::cout << "Actual partition time: \033[1;33m" << timer[0] << "\033[0m" << std::endl;
    
    // the lookup function, taking vertices to aggregates
    auto find = [&] (int i) {
      int root = i;
      while (id[root] != root) {
	root = id[root];
      }
      while (id[i] != root) {
	int a = id[i];
	id[i] = root;
	i = a;
      }
      return root;
    };

    // create the partition
    std::vector<std::vector<int>> partition (M);
    for (int y=0; y<N /* == basis.size()*/; y++) {
      int i = y; //basis[y];
      int root = find(i);
      int index = pointer[root];
      partition[index].push_back(y);
    }
    
    return interpolationMatrix(N, partition);
      
  }


  SparseMatrix partitionBase (const SparseMatrix& A, const float stallStopThreshold) {

    // at any iteration, if we created a partition it would have M rows and N columns
    int N = A.Rows(); 
    int M = N;
  
    std::vector<float> alpha (N);
    
    const std::vector<int>& I = A.GetIndptr();
    const std::vector<int>& J = A.GetIndices();
    const std::vector<double>& D = A.GetData();
  
    // store edge weights in a vector of trees
    std::vector<std::map<int, float> > a (N);

    #pragma omp parallel for
    for (int i=0; i<N; i++) {
      float alpha_i = 0.0;
      for (int k=I[i]; k<I[i+1]; k++) {
	int j = J[k];
	float a_ij = D[k];
	if (i != j) {
	  a[i].insert(std::make_pair (j, a_ij));
	}

	alpha_i += a_ij;
 
      }
      alpha[i] = alpha_i;
    }

    float d_sum = 0.0;
    float T = 0.0;
    for (int i=0; i<N; i++) {
      for (int k=I[i]; k<I[i+1]; k++) {
	int j = J[k];
	float a_ij = D[k];
	if (i == j) {
	  d_sum += a_ij;
	} 
	T += a_ij;
      }
    }

    #pragma omp parallel for
    for (int i=0; i<alpha.size(); i++) {
      alpha[i] /= T;
    }


    float inf = std::numeric_limits<float>::infinity();
  
    float sum = d_sum / T;
    for (int i=0; i<N; i++) {
      sum += - alpha[i] * alpha[i];
    }
    // initial Q
    // \sum_{i} d_i / T - \alpha_i^2
    float Q = sum; 
    
    // this vector tells us what vertices still exist in the graph after we create a partition
    // for creating a single partition, this vector won't be touched, so be don't need to create it
    // std::vector<int> basis (N);


    // used is a list of the vertices that are still in use
    std::vector<int> used (N);
    // pointer[i] is the index of i in used
    std::vector<int> pointer (N);

    // this is the disjoint sets data structure
    std::vector<int> id (N);


    #pragma omp parallel for
    for (int i=0; i<N; i++) {
      //basis[i] = i;
      used[i] = i;
      pointer[i] = i;
      id[i] = i;
    }

    // stores the maximum \eta (\Delta \Q) value
    std::vector<float> max_eta (N, -inf);
    // stores the best index to merge with
    std::vector<float> max_ind (N, -1);
    // stores if given vertex has been merged during an iteration
    std::vector<bool> notouch (N, false);

    M = used.size();
    int M_prev = M;

    linalgcpp::Timer timer (linalgcpp::Timer::Start::True);
    do {    
      assert(M == used.size());
      //assert(N == basis.size());

      // keeps track of the merged pairs    
      std::vector<std::pair<int, int>> merged;
      float DeltaQ = 0.0;

      #pragma omp parallel for
      for (int x=0; x<used.size(); x++) {
	int i = used[x];
	if (!notouch[i] || max_eta[i] == -inf) {
	  float max_eta_i = -inf;
	  int max_coord = -1;

	  // find the best one to merge with that hasn't been merged
	  for (const auto& kv : a[i]) {
	    int j = kv.first;
	    if (!notouch[j]) {
	      float a_ij = kv.second;
	      assert(i != j);
	      float eta = 2 * (a_ij / T - alpha[i] * alpha[j]); 
	      if (eta > max_eta_i) {
		max_eta_i = eta;
		max_coord = j;
	      }
	    }
	  }
	  
	  max_eta[i] = max_eta_i;
	  max_ind[i] = max_coord;
	}
      }

      for (int x=0; x<used.size(); x++) {
	int i = used[x];
	if (!notouch[i]) {
	  int j = max_ind[i];
	  // If the score of the one to merge with is no better than this nodes score, then merge
	  if (j != -1 && !notouch[j] && max_eta[i] >= max_eta[j]) {
	    float DeltaQ_ij = max_eta[i];
	    
	    if (DeltaQ_ij > 0) {
	      if (i < j) {
		int i_prime, j_prime;
		if (a[i].size() < a[j].size()) {
		  i_prime = j;
		  j_prime = i;
		} else {
		  i_prime = i;
		  j_prime = j;
		}
	      
		merged.push_back(std::make_pair(i_prime,j_prime));
	      
		notouch[i] = true;
		notouch[j] = true;

		DeltaQ += DeltaQ_ij;
	      }
	    }
	  }
	}
      }
      
      // the "matrix multiplication" step, merging eges
      for (int x=0; x<merged.size(); x++) {
	int i_prime = std::get<0>(merged[x]);
	int j_prime = std::get<1>(merged[x]);

	// i_prime is the new representative, j_prime is being removed
	// for every edge connecting j_prime and k, we have add or update the edge connecting i_prime and k
	for (const auto& elt : a[j_prime]) { 
	  int k = elt.first; 
	  float a_jk = elt.second;
	  a[k].erase(a[k].find(j_prime));

	  max_eta[k] = -inf;
	
	  if (k == i_prime) {
	    //float a_ij = a_jk;
	    //d[i_prime] = d[i_prime] + d[j_prime] + 2*a_ij;
	    alpha[i_prime] = alpha[i_prime] + alpha[j_prime];
	  } else {
	    a[i_prime][k] += a_jk;
	    a[k][i_prime] += a_jk;
	  
	  }
	
	}
      }

      {
	std::vector<int> test(N, 0);
	for (int x=0; x<merged.size(); x++) {
	  int i_prime = std::get<0>(merged[x]);
	  for (const auto& elt : a[i_prime]) { 
	    int k = elt.first; 
	    test[k] = 1;
	  }
	}
	int sum = 0;
	for (int i=0; i<test.size(); i++) {
	  sum += test[i];
	}
	//std::cout << "fraction touched: " << (1.0 * sum) / used.size() << std::endl;
      }

      // ===========

      Q += DeltaQ;

      M_prev = M;

      // remove all j_prime from used, and update the pointers
      for (int x=0; x<merged.size(); x++) {
	int i_prime = std::get<0>(merged[x]);
	int j_prime = std::get<1>(merged[x]);
	int index_j_prime = pointer[j_prime];
	int k = used[used.size() -1 ];
	std::iter_swap(used.begin() + index_j_prime, used.end() -1);
	used.pop_back();
	pointer[k] = index_j_prime;

	id[j_prime] = i_prime;
    
	notouch[i_prime] = false;
      }

      M -= merged.size();
      //std::cout << "N: " << M << " | " << (1.0 * M) / M_prev << std::endl;

	//std::cout << std::setprecision(2) << "ratio: " << (1.0 * M)/M_prev << "   \t Q: " << Q << "   \t |{A}|: " << M << std::endl; //<< "   \t avg |a_i|: " << (1.0 * sum) / used.size() << std::endl;

    } // continue until the coarsening factor is large enough; by default only halt if M == M_prev
    while (1.0*M/M_prev < stallStopThreshold);

    timer.Click();
    std::cout << "Actual partition time: \033[1;33m" << timer[0] << "\033[0m" << std::endl;
    
    // the lookup function, taking vertices to aggregates
    auto find = [&] (int i) {
      int root = i;
      while (id[root] != root) {
	root = id[root];
      }
      while (id[i] != root) {
	int a = id[i];
	id[i] = root;
	i = a;
      }
      return root;
    };

    // create the partition
    std::vector<std::vector<int>> partition (M);
    for (int y=0; y<N /* == basis.size()*/; y++) {
      int i = y; //basis[y];
      int root = find(i);
      int index = pointer[root];
      partition[index].push_back(y);
    }
    
    return interpolationMatrix(N, partition);
      
  }

  SparseMatrix partitionBase2 (const SparseMatrix& A, const float stallStopThreshold) {

    // at any iteration, if we created a partition it would have M rows and N columns
    int N = A.Rows(); 
    int M = N;
  
    std::vector<float> alpha (N);
    
    const std::vector<int>& I = A.GetIndptr();
    const std::vector<int>& J = A.GetIndices();
    const std::vector<double>& D = A.GetData();
  
    // store edge weights in a vector of trees
    std::vector<std::vector<int>> a (N);
    std::vector<std::vector<float>> d (N);

    #pragma omp parallel for
    for (int i=0; i<N; i++) {
      float alpha_i = 0.0;
      for (int k=I[i]; k<I[i+1]; k++) {
	int j = J[k];
	float a_ij = D[k];
	if (i != j) {
	  a[i].push_back(j);
	  d[i].push_back(a_ij);
	}

	alpha_i += a_ij;
 
      }
      alpha[i] = alpha_i;
    }


    float d_sum = 0.0;
    float T = 0.0;
    for (int i=0; i<N; i++) {
      for (int k=I[i]; k<I[i+1]; k++) {
	int j = J[k];
	float a_ij = D[k];
	if (i == j) {
	  d_sum += a_ij;
	} 
	T += a_ij;
      }
    }

    //double T = 0.0;
    //for (int i=0; i<N; i++) {
    //  T += alpha[i];
    //}

    #pragma omp for
    for (int i=0; i<alpha.size(); i++) {
      alpha[i] /= T;
    }


    float inf = std::numeric_limits<double>::infinity();
  
    float sum = d_sum / T;
    for (int i=0; i<N; i++) {
      sum += - alpha[i] * alpha[i];
    }
    // initial Q
    // \sum_{i} d_i / T - \alpha_i^2
    float Q = sum; 
    
    // this vector tells us what vertices still exist in the graph after we create a partition
    // for creating a single partition, this vector won't be touched, so be don't need to create it
    // std::vector<int> basis (N);


    // used is a list of the vertices that are still in use
    std::vector<int> used (N);
    // pointer[i] is the index of i in used
    std::vector<int> pointer (N);

    // this is the disjoint sets data structure
    std::vector<int> id (N);


    #pragma omp parallel for
    for (int i=0; i<N; i++) {
      //basis[i] = i;
      used[i] = i;
      pointer[i] = i;
      id[i] = i;
    }

    // stores the maximum \eta (\Delta \Q) value
    std::vector<float> max_eta (N, -inf);
    // stores the best index to merge with
    std::vector<float> max_ind (N);
    // stores if given vertex has been merged during an iteration
    std::vector<bool> notouch (N, false);

    M = used.size();
    int M_prev = M;

    linalgcpp::Timer timer (linalgcpp::Timer::Start::True);
    do {    
      assert(M == used.size());
      //assert(N == basis.size());

      // keeps track of the merged pairs    
      std::vector<std::pair<int, int>> merged;
      float DeltaQ = 0.0;

      #pragma omp parallel for
      for (int x=0; x<used.size(); x++) {
	int i = used[x];
	if (!notouch[i] || max_eta[i] == -inf) {
	  float max_eta_i = -inf;
	  int max_coord = -1;

	  // find the best one to merge with that hasn't been merged
	  for (int y=0; y<a[i].size(); y++) {
	    int j = a[i][y];
	    if (!notouch[j]) {
	      float a_ij = d[i][y];
	      assert(i != j);
	      float eta = 2 * (a_ij / T - alpha[i] * alpha[j]); 
	      if (eta > max_eta_i) {
		max_eta_i = eta;
		max_coord = j;
	      }
	    }
	  }
	  
	  max_eta[i] = max_eta_i;
	  max_ind[i] = max_coord;
	}
      }

      for (int x=0; x<used.size(); x++) {
	int i = used[x];
	if (!notouch[i]) {
	  int j = max_ind[i];
	  // If the score of the one to merge with is no better than this nodes score, then merge
	  if (j != -1 && !notouch[j] && !(max_eta[i] < max_eta[j])) {
	    float DeltaQ_ij = max_eta[i];
	    
	    if (DeltaQ_ij > 0) {
	      int i_prime, j_prime;
	      if (a[i].size() < a[j].size()) {
		i_prime = j;
		j_prime = i;
	      } else {
		i_prime = i;
		j_prime = j;
	      }
	      merged.push_back(std::make_pair(i_prime,j_prime));
	      
	      notouch[i] = true;
	      notouch[j] = true;
	      
	      DeltaQ += DeltaQ_ij;
	    }
	  }
	}
      }

      // the "matrix multiplication" step, merging eges
      for (int x=0; x<merged.size(); x++) {
	int i_prime = std::get<0>(merged[x]);
	int j_prime = std::get<1>(merged[x]);

	// i_prime is the new representative, j_prime is being removed
	// for every edge connecting j_prime and k, we have add or update the edge connecting i_prime and k
	for (int y=0; y<a[j_prime].size(); y++) {
	  int k = a[j_prime][y]; 
	  float a_jk = d[j_prime][y];

	  for (int z=0; z<a[k].size(); z++) {
	    if (a[k][z] == j_prime) {
	      int size = a[k].size()-1;
	      int temp = a[k][size];
	      float temp2 = d[k][size];

	      a[k][size] = j_prime;
	      d[k][size] = a_jk;

	      a[k][z] = temp;
	      d[k][z] = temp2;

	      a[k].pop_back();
	      d[k].pop_back();

	      break;
	    }
	  }

	  max_eta[k] = -inf;
	
	  if (k == i_prime) {
	    //float a_ij = a_jk;
	    //d[i_prime] = d[i_prime] + d[j_prime] + 2*a_ij;
	    alpha[i_prime] = alpha[i_prime] + alpha[j_prime];
	  } else {
	    for (int z=0; z<a[i_prime].size(); z++) {
	      if (a[i_prime][z] == k) {
		d[i_prime][z] += a_jk;
	      }
	    }
	    for (int z=0; z<a[k].size(); z++) {
	      if (a[k][z] == i_prime) {
		d[k][z] += a_jk;
	      }
	    }
	  }
	
	}
      }
      // ===========

      Q += DeltaQ;

      M_prev = M;

      // remove all j_prime from used, and update the pointers
      for (int x=0; x<merged.size(); x++) {
	int i_prime = std::get<0>(merged[x]);
	int j_prime = std::get<1>(merged[x]);
	int index_j_prime = pointer[j_prime];
	int k = used[used.size() -1 ];
	std::iter_swap(used.begin() + index_j_prime, used.end() -1);
	used.pop_back();
	pointer[k] = index_j_prime;

	id[j_prime] = i_prime;
    
	notouch[i_prime] = false;
      }

      M -= merged.size();

	//std::cout << std::setprecision(2) << "ratio: " << (1.0 * M)/M_prev << "   \t Q: " << Q << "   \t |{A}|: " << M << std::endl; //<< "   \t avg |a_i|: " << (1.0 * sum) / used.size() << std::endl;

    } // continue until the coarsening factor is large enough; by default only halt if M == M_prev
    while (1.0*M/M_prev < stallStopThreshold);

    timer.Click();
    std::cout << "Actual partition time: \033[1;33m" << timer[0] << "\033[0m" << std::endl;
    
    // the lookup function, taking vertices to aggregates
    auto find = [&] (int i) {
      int root = i;
      while (id[root] != root) {
	root = id[root];
      }
      while (id[i] != root) {
	int a = id[i];
	id[i] = root;
	i = a;
      }
      return root;
    };

    // create the partition
    std::vector<std::vector<int>> partition (M);
    for (int y=0; y<N /* == basis.size()*/; y++) {
      int i = y; //basis[y];
      int root = find(i);
      int index = pointer[root];
      partition[index].push_back(y);
    }
    
    return interpolationMatrix(N, partition);
      
  }

  

  SparseMatrix partition (const SparseMatrix& A, const bool printing, const bool positiveMerging, const double stallStopThreshold, const int matchingIterations, const bool mergeLeaves) {
  
    // at any iteration, if we created a partition it would have M rows and N columns
    int N = A.Rows(); 
    int M = N;
  
    std::vector<double> alpha (N);
    
    const std::vector<int>& I = A.GetIndptr();
    const std::vector<int>& J = A.GetIndices();
    const std::vector<double>& D = A.GetData();
  
    // store edge weights in a vector of trees
    std::vector<std::map<int, double> > a (N);

    #pragma omp parallel for
    for (int i=0; i<N; i++) {
      double alpha_i = 0.0;
      for (int k=I[i]; k<I[i+1]; k++) {
	int j = J[k];
	double a_ij = D[k];
	if (i != j) {
	  a[i].insert(std::make_pair (j, a_ij));
	}

	alpha_i += a_ij;
 
      }
      alpha[i] = alpha_i;
    }

    double d_sum = 0.0;
    double T = 0.0;
    for (int i=0; i<N; i++) {
      for (int k=I[i]; k<I[i+1]; k++) {
	int j = J[k];
	double a_ij = D[k];
	if (i == j) {
	  d_sum += a_ij;
	} 
	T += a_ij;
      }
    }

    #pragma omp for
    for (int i=0; i<alpha.size(); i++) {
      alpha[i] /= T;
    }


    double inf = std::numeric_limits<double>::infinity();
  
    double sum = d_sum / T;
    for (int i=0; i<N; i++) {
      sum += - alpha[i] * alpha[i];
    }
    // initial Q
    // \sum_{i} d_i / T - \alpha_i^2
    double Q = sum; 
    
    // this vector tells us what vertices still exist in the graph after we create a partition
    // for creating a single partition, this vector won't be touched, so be don't need to create it
    // std::vector<int> basis (N);


    // used keeps track of what vertices are still in use
    std::vector<int> used (N);
    // pointer[i] is the index of i in used
    std::vector<int> pointer (N);

    // this is the disjoint sets data structure
    std::vector<int> id (N);


    #pragma omp parallel for
    for (int i=0; i<N; i++) {
      //basis[i] = i;
      used[i] = i;
      pointer[i] = i;
      id[i] = i;
    }

    // the lookup function, taking vertices to aggregates
    auto find = [&] (int i) {
      int root = i;
      while (id[root] != root) {
	root = id[root];
      }
      while (id[i] != root) {
	int a = id[i];
	id[i] = root;
	i = a;
      }
      return root;
    };

    // stores the maximum \eta (\Delta \Q) value
    std::vector<double> max_eta (N, -inf);
    // stores the best index to merge with
    std::vector<double> max_ind (N);
    // stores if given vertex has been merged during an iteration
    std::vector<bool> notouch (N, false);

    // merge leaves
    bool changed = true;
    while (mergeLeaves && changed) {
      changed = false;
      for (int x=0; x<used.size(); x++) {
	int i = used[x];
	if (a[i].size() == 1) {  
	  int leaf = i;

	  int notleaf;
	  double a_leafnotleaf;

	  for (const auto& entry : a[i]) {
	    notleaf = entry.first;
	    a_leafnotleaf = entry.second;
	  }

	  double DeltaQ = 2*(a_leafnotleaf/T - alpha[leaf] * alpha[notleaf]);
	  if (!positiveMerging || DeltaQ > 0) {
	    changed = true;
	    Q += DeltaQ;
	  
	    id[leaf] = notleaf;
	    //d[notleaf] = d[notleaf] + d[leaf] + 2*a_leafnotleaf;
	    alpha[notleaf] = alpha[notleaf] + alpha[leaf];
	    // remove leaf's edge from notleaf
	    a[notleaf].erase(a[notleaf].find(leaf));
	  
	  
	  
	    int index_leaf = pointer[leaf];
	    int k = used[used.size() -1 ];
	    
	    int tmp = used[index_leaf];
	    used[index_leaf] = used[used.size()-1];
	    used[used.size() - 1] = index_leaf;

	    pointer[k] = index_leaf;
	    used.pop_back();
	  
	  }
	}
      }
    }

    M = used.size();
    int M_prev = M;

    int num_iter = 0;
    // iterate

    linalgcpp::Timer timer (linalgcpp::Timer::Start::True);
    do {
      num_iter++;

      assert(M == used.size());
      //assert(N == basis.size());

      // keeps track of the merged pairs    
      std::vector<std::pair<int, int>> merged;
      double DeltaQ = 0.0;
    
      for (int iter=0; iter<matchingIterations; iter++) {

	#pragma omp parallel for
	for (int x=0; x<used.size(); x++) {
	  int i = used[x];
	  if (!notouch[i] || max_eta[i] == -inf) {
	    double max_eta_i = -inf;
	    int max_coord = -1;

	    // find the best one to merge with that hasn't been merged
	    for (const auto& kv : a[i]) {
	      int j = kv.first;
	      if (!notouch[j]) {
		double a_ij = kv.second;
		assert(i != j);
		double eta = 2 * (a_ij / T - alpha[i] * alpha[j]); 
		if (eta > max_eta_i) {
		  max_eta_i = eta;
		  max_coord = j;
		}
	      }
	    }
	  
	    max_eta[i] = max_eta_i;
	    max_ind[i] = max_coord;
	  }
	}
      
	for (int x=0; x<used.size(); x++) {
	  int i = used[x];
	  if (!notouch[i]) {
	    int j = max_ind[i];
	    // If the score of the one to merge with is no better than this nodes score, then merge
	    if (j != -1 && !notouch[j] && !(max_eta[i] < max_eta[j])) {
	      double DeltaQ_ij = max_eta[i];
	    
	      if (!positiveMerging || DeltaQ_ij > 0) {
		int i_prime, j_prime;
		if (a[i].size() < a[j].size()) {
		  i_prime = j;
		  j_prime = i;
		} else {
		  i_prime = i;
		  j_prime = j;
		}
		merged.push_back(std::make_pair(i_prime,j_prime));
	      
		notouch[i] = true;
		notouch[j] = true;
	      
		DeltaQ += DeltaQ_ij;
	      }
	    }
	  }
	}
      }

      // the "matrix multiplication" step, merging eges
      for (int x=0; x<merged.size(); x++) {
	int i_prime = std::get<0>(merged[x]);
	int j_prime = std::get<1>(merged[x]);

	// i_prime is the new representative, j_prime is being removed
	// for every edge connecting j_prime and k, we have add or update the edge connecting i_prime and k
	for (const auto& elt : a[j_prime]) { 
	  int k = elt.first; 
	  double a_jk = elt.second;
	  a[k].erase(a[k].find(j_prime));

	  max_eta[k] = -inf;
	
	  if (k == i_prime) {
	    //double a_ij = a_jk;
	    //d[i_prime] = d[i_prime] + d[j_prime] + 2*a_ij;
	    alpha[i_prime] = alpha[i_prime] + alpha[j_prime];
	  } else {
	    a[i_prime][k] += a_jk;
	    a[k][i_prime] += a_jk;
	  
	  }
	
	}

      } 
      // ===========

      Q += DeltaQ;

      M_prev = M;

      // remove all j_prime from used, and update the pointers
      for (int x=0; x<merged.size(); x++) {
	int i_prime = std::get<0>(merged[x]);
	int j_prime = std::get<1>(merged[x]);
	int index_j_prime = pointer[j_prime];
	int k = used[used.size() -1 ];
	std::iter_swap(used.begin() + index_j_prime, used.end() -1);
	used.pop_back();
	pointer[k] = index_j_prime;

	id[j_prime] = i_prime;
    
	notouch[i_prime] = false;
      }

      M -= merged.size();

      //std::cout << std::setprecision(2) << "ratio: " << (1.0 * M)/M_prev << "   \t Q: " << Q << "   \t |{A}|: " << M << std::endl; //<< "   \t avg |a_i|: " << (1.0 * sum) / used.size() << std::endl;

    } // continue until the coarsening factor is large enough; by default only halt if M == M_prev
    while (1.0*M/M_prev < stallStopThreshold);

    timer.Click();
    std::cout << "Actual partition time: \033[1;33m" << timer[0] << "\033[0m" << std::endl;

    // create the partition
    std::vector<std::vector<int>> partition (M);
    for (int y=0; y<N /* == basis.size()*/; y++) {
      int i = y; //basis[y];
      int root = find(i);
      int index = pointer[root];
      partition[index].push_back(y);
    }
    
    if (printing) {
      std::cout << "modularity: " << Q << std::endl;
      std::cout << "aggregates: " << M << std::endl;
    }
  
    return interpolationMatrix(N, partition);
      
  }





  SparseMatrix partition (const SparseMatrix& A, const int numParts, const bool printing, const bool positiveMerging, const double stallStopThreshold, const int matchingIterations, const bool mergeLeaves) {
  
    int N = A.Rows(); 
    int M = N;
  
    std::vector<double> alpha (N);
    
    const std::vector<int>& I = A.GetIndptr();
    const std::vector<int>& J = A.GetIndices();
    const std::vector<double>& D = A.GetData();
  
    std::vector<std::map<int, double> > a (N);

    #pragma omp parallel for
    for (int i=0; i<N; i++) {
      double alpha_i = 0.0;
      for (int k=I[i]; k<I[i+1]; k++) {
	int j = J[k];
	double a_ij = D[k];
	if (i != j) {
	  a[i].insert(std::make_pair (j, a_ij));
	}

	alpha_i += a_ij;
 
      }
      alpha[i] = alpha_i;
    }

    double d_sum = 0.0;
    double T = 0.0;
    for (int i=0; i<N; i++) {
      for (int k=I[i]; k<I[i+1]; k++) {
	int j = J[k];
	double a_ij = D[k];
	if (i == j) {
	  d_sum += a_ij;
	} 
	T += a_ij;
      }
    }

    #pragma omp for
    for (int i=0; i<alpha.size(); i++) {
      alpha[i] /= T;
    }


    double inf = std::numeric_limits<double>::infinity();
  
    double sum = d_sum / T;
    for (int i=0; i<N; i++) {
      sum += - alpha[i] * alpha[i];
    }
    double Q = sum; // \sum_{i} d_i / T - \alpha_i^2
    
    //std::vector<int> basis (N);

    std::vector<int> used (N);
    std::vector<int> pointer (N);

    std::vector<int> id (N);


    #pragma omp parallel for
    for (int i=0; i<N; i++) {
      //basis[i] = i;
      used[i] = i;
      pointer[i] = i;
      id[i] = i;
    }

    auto find = [&] (int i) {
      int root = i;
      while (id[root] != root) {
	root = id[root];
      }
      while (id[i] != root) {
	int a = id[i];
	id[i] = root;
	i = a;
      }
      return root;
    };

    std::vector<double> max_eta (N, -inf);
    std::vector<double> max_ind (N);
    std::vector<bool> notouch (N, false);

    std::vector<SparseMatrix> Ps;

    // merge leaves
    bool changed = true;
    while (mergeLeaves && changed) {
      changed = false;
      for (int x=0; x<used.size(); x++) {
	int i = used[x];
	if (a[i].size() == 1) {  
	  int leaf = i;

	  int notleaf;
	  double a_leafnotleaf;

	  for (const auto& entry : a[i]) {
	    notleaf = entry.first;
	    a_leafnotleaf = entry.second;
	  }

	  double DeltaQ = 2*(a_leafnotleaf/T - alpha[leaf] * alpha[notleaf]);
	  if (!positiveMerging || DeltaQ > 0) {
	    changed = true;
	    Q += DeltaQ;
	  
	    id[leaf] = notleaf;
	    //d[notleaf] = d[notleaf] + d[leaf] + 2*a_leafnotleaf;
	    alpha[notleaf] = alpha[notleaf] + alpha[leaf];
	    // remove leaf's edge from notleaf...
	    a[notleaf].erase(a[notleaf].find(leaf));
	  
	  
	  
	    int index_leaf = pointer[leaf];
	    int k = used[used.size() -1 ];
	    //std::iter_swap(used.begin() + index_leaf, used.end() -1);
	    
	    int tmp = used[index_leaf];
	    used[index_leaf] = used[used.size()-1];
	    used[used.size() - 1] = index_leaf;

	    pointer[k] = index_leaf;
	    used.pop_back();
	  
	    M -= 1;
	  }
	}
      }
    }

    M = used.size();
    int M_prev = M;

    int num_iter = 0;
    // iterate

    do {
      num_iter++;
    
      std::vector<std::pair<int, int>> merged;
      double DeltaQ = 0.0;
    
      for (int iter=0; iter<matchingIterations; iter++) {
	#pragma omp parallel for
	for (int x=0; x<used.size(); x++) {
	  int i = used[x];
	  if (!notouch[i] || max_eta[i] == -inf) {
	    double max_eta_i = -inf;
	    int max_coord = -1;

	    for (const auto& kv : a[i]) {
	      int j = kv.first;
	      if (!notouch[j]) {
		double a_ij = kv.second;
		assert(i != j);
		double eta = 2 * (a_ij / T - alpha[i] * alpha[j]); 
		if (eta > max_eta_i) {
		  max_eta_i = eta;
		  max_coord = j;
		}
	      }
	    }
	  
	    max_eta[i] = max_eta_i;
	    max_ind[i] = max_coord;
	  }
	}
      
	for (int x=0; x<used.size(); x++) {
	  int i = used[x];
	  if (!notouch[i]) {
	    int j = max_ind[i];
	    if (j != -1 && !notouch[j] && !(max_eta[i] < max_eta[j])) {
	      double DeltaQ_ij = max_eta[i];
	    
	      if (!positiveMerging || DeltaQ_ij > 0) {
		int i_prime, j_prime;
		if (a[i].size() < a[j].size()) {
		  i_prime = j;
		  j_prime = i;
		} else {
		  i_prime = i;
		  j_prime = j;
		}
		merged.push_back(std::make_pair(i_prime,j_prime));
	      
		notouch[i] = true;
		notouch[j] = true;
	      
		DeltaQ += DeltaQ_ij;
	      }
	    }
	  }
	}
      }

      for (int x=0; x<merged.size(); x++) {
	int i_prime = std::get<0>(merged[x]);
	int j_prime = std::get<1>(merged[x]);

	for (const auto& elt : a[j_prime]) { //(auto it = a[j_prime].begin(); it != a[j_prime].end(); it++) {
	  int k = elt.first; //it->first;
	  double a_jk = elt.second; //it->second;
	  a[k].erase(a[k].find(j_prime));

	  max_eta[k] = -inf;
	
	  if (k == i_prime) {
	    //double a_ij = a_jk;
	    //d[i_prime] = d[i_prime] + d[j_prime] + 2*a_ij;
	    alpha[i_prime] = alpha[i_prime] + alpha[j_prime];
	  } else {
	    a[i_prime][k] += a_jk;
	    a[k][i_prime] += a_jk;
	  
	  }
	
	}

      } 
      // ===========

      Q += DeltaQ;
    
      M_prev = M;


      for (int x=0; x<merged.size(); x++) {
	int i_prime = std::get<0>(merged[x]);
	int j_prime = std::get<1>(merged[x]);
	int index_j_prime = pointer[j_prime];
	int k = used[used.size() -1 ];
	std::iter_swap(used.begin() + index_j_prime, used.end() -1);
	used.pop_back();
	pointer[k] = index_j_prime;

	id[j_prime] = i_prime;
    
	notouch[i_prime] = false;
      }
      
      M -= merged.size();

      //std::cout << std::setprecision(2) << "ratio: " << (1.0 * M)/M_prev << "   \t Q: " << Q << "   \t |{A}|: " << M << std::endl; //<< "   \t avg |a_i|: " << (1.0 * sum) / used.size() << std::endl;

    } while (1.0*M/M_prev < stallStopThreshold && M > numParts);

    
    std::vector<std::vector<int>> partition (M);
    for (int y=0; y<N /* == basis.size()*/; y++) {
      int i = y; //basis[y];
      int root = find(i);
      int index = pointer[root];
      partition[index].push_back(y);
    }
     

    if (printing) {
      std::cout << "modularity: " << Q << std::endl;
      std::cout << "aggregates: " << M << std::endl;
    }
  
    return interpolationMatrix(N, partition);
      
  }





  std::vector<SparseMatrix> partition (const SparseMatrix& A, const double coarseningFactor, const bool printing, const bool positiveMerging, const double stallStopThreshold, const int matchingIterations, const bool mergeLeaves) {
  
    int N = A.Rows(); 
    int M = N;
  
    std::vector<double> alpha (N);
    
    const std::vector<int>& I = A.GetIndptr();
    const std::vector<int>& J = A.GetIndices();
    const std::vector<double>& D = A.GetData();
  
    std::vector<std::map<int, double> > a (N);

    #pragma omp parallel for
    for (int i=0; i<N; i++) {
      double alpha_i = 0.0;
      for (int k=I[i]; k<I[i+1]; k++) {
	int j = J[k];
	double a_ij = D[k];
	if (i != j) {
	  a[i].insert(std::make_pair (j, a_ij));
	}

	alpha_i += a_ij;
 
      }
      alpha[i] = alpha_i;
    }


    double d_sum = 0.0;
    double T = 0.0;
    for (int i=0; i<N; i++) {
      for (int k=I[i]; k<I[i+1]; k++) {
	int j = J[k];
	double a_ij = D[k];
	if (i == j) {
	  d_sum += a_ij;
	} 
	T += a_ij;
      }
    }

    #pragma omp for
    for (int i=0; i<alpha.size(); i++) {
      alpha[i] /= T;
    }


    double inf = std::numeric_limits<double>::infinity();
  
    double sum = d_sum / T;
    for (int i=0; i<N; i++) {
      sum += - alpha[i] * alpha[i];
    }
    double Q = sum; // \sum_{i} d_i / T - \alpha_i^2
    double Q_old = Q;
    double Q_max = Q;
    bool increasing_Q = true;

    std::vector<int> basis (N);

    #pragma omp parallel for
    for (int i=0; i<N; i++) {
      basis[i] = i;
    }

    std::vector<int> used = basis;
    std::vector<int> pointer = basis;

    std::vector<int> id = basis;

    auto find = [&] (int i) {
      int root = i;
      while (id[root] != root) {
	root = id[root];
      }
      while (id[i] != root) {
	int a = id[i];
	id[i] = root;
	i = a;
      }
      return root;
    };

    std::vector<double> max_eta (N, -inf);
    std::vector<double> max_ind (N);
    std::vector<bool> notouch (N, false);

    std::vector<SparseMatrix> Ps;

    // merge leaves
    bool changed = true;
    while (mergeLeaves && changed) {
      assert(M == used.size());
      assert(N == basis.size());

      changed = false;
      for (int x=0; x<used.size(); x++) {
	int i = used[x];
	if (a[i].size() == 1) {  
	  int leaf = i;

	  int notleaf;
	  double a_leafnotleaf;

	  for (const auto& entry : a[i]) {
	    notleaf = entry.first;
	    a_leafnotleaf = entry.second;
	  }

	  double DeltaQ = 2*(a_leafnotleaf/T - alpha[leaf] * alpha[notleaf]);
	  if (!positiveMerging || DeltaQ > 0) {
	    changed = true;
	    Q += DeltaQ;
	  
	    id[leaf] = notleaf;
	    //d[notleaf] = d[notleaf] + d[leaf] + 2*a_leafnotleaf;
	    alpha[notleaf] = alpha[notleaf] + alpha[leaf];
	    // remove leaf's edge from notleaf...
	    a[notleaf].erase(a[notleaf].find(leaf));
	  
	  
	  
	    int index_leaf = pointer[leaf];
	    int k = used[used.size() -1 ];
	    //std::iter_swap(used.begin() + index_leaf, used.end() -1);
	    
	    int tmp = used[index_leaf];
	    used[index_leaf] = used[used.size()-1];
	    used[used.size() - 1] = index_leaf;

	    pointer[k] = index_leaf;
	    used.pop_back();
	  
	    M -= 1;
	  }
	}
      }
    }

    int M_prev = M;

    int num_iter = 0;
    // iterate

    do {
      num_iter++;
    
      std::vector<std::pair<int, int>> merged;
      double DeltaQ = 0.0;
    
      for (int iter=0; iter<matchingIterations; iter++) {
	#pragma omp parallel for
	for (int x=0; x<used.size(); x++) {
	  int i = used[x];
	  if (!notouch[i] || max_eta[i] == -inf) {
	    double max_eta_i = -inf;
	    int max_coord = -1;

	    for (const auto& kv : a[i]) {
	      int j = kv.first;
	      if (!notouch[j]) {
		double a_ij = kv.second;
		assert(i != j);
		double eta = 2 * (a_ij / T - alpha[i] * alpha[j]); 
		if (eta > max_eta_i) {
		  max_eta_i = eta;
		  max_coord = j;
		}
	      }
	    }
	  
	    max_eta[i] = max_eta_i;
	    max_ind[i] = max_coord;
	  }
	}
      
	for (int x=0; x<used.size(); x++) {
	  int i = used[x];
	  if (!notouch[i]) {
	    int j = max_ind[i];
	    if (j != -1 && !notouch[j] && !(max_eta[i] < max_eta[j])) {
	      double DeltaQ_ij = max_eta[i];
	    
	      if (!positiveMerging || DeltaQ_ij > 0) {
		int i_prime, j_prime;
		if (a[i].size() < a[j].size()) {
		  i_prime = j;
		  j_prime = i;
		} else {
		  i_prime = i;
		  j_prime = j;
		}
		merged.push_back(std::make_pair(i_prime,j_prime));
	      
		notouch[i] = true;
		notouch[j] = true;
	      
		DeltaQ += DeltaQ_ij;
	      }
	    }
	  }
	}
      }

      for (int x=0; x<merged.size(); x++) {
	int i_prime = std::get<0>(merged[x]);
	int j_prime = std::get<1>(merged[x]);

	for (const auto& elt : a[j_prime]) { //(auto it = a[j_prime].begin(); it != a[j_prime].end(); it++) {
	  int k = elt.first; //it->first;
	  double a_jk = elt.second; //it->second;
	  a[k].erase(a[k].find(j_prime));

	  max_eta[k] = -inf;
	
	  if (k == i_prime) {
	    //double a_ij = a_jk;
	    //d[i_prime] = d[i_prime] + d[j_prime] + 2*a_ij;
	    alpha[i_prime] = alpha[i_prime] + alpha[j_prime];
	  } else {
	    a[i_prime][k] += a_jk;
	    a[k][i_prime] += a_jk;
	  
	  }
	
	}

      } 
      // ===========

      Q_old = Q;
    
      Q += DeltaQ;

      if (Q_old<Q) {
	increasing_Q = true;
      }
    
      if (Q_max < Q) {
	Q_max = Q;
      }
    
      M_prev = M;


      if (1.0 * M / N <= coarseningFactor) {
      
	std::vector<std::vector<int>> partition (M);
	
	for (int y=0; y<basis.size(); y++) {
	  int i = basis[y];
	  int root = find(i);
	  int index = pointer[root];
	  partition[index].push_back(y);
	}
		
	Ps.push_back(interpolationMatrix(N, partition));
      
	basis = used;
	N = M;
    
	increasing_Q = false;
      
      }   



      for (int x=0; x<merged.size(); x++) {
	int i_prime = std::get<0>(merged[x]);
	int j_prime = std::get<1>(merged[x]);
	int index_j_prime = pointer[j_prime];
	int k = used[used.size() -1 ];
	std::iter_swap(used.begin() + index_j_prime, used.end() -1);
	used.pop_back();
	pointer[k] = index_j_prime;

	id[j_prime] = i_prime;
    
	notouch[i_prime] = false;
      
	M -= 1;

      }

      //std::cout << std::setprecision(2) << "ratio: " << (1.0 * M)/M_prev << "   \t Q: " << Q << "   \t |{A}|: " << M << std::endl; //<< "   \t avg |a_i|: " << (1.0 * sum) / used.size() << std::endl;

    } while (1.0*M/M_prev < stallStopThreshold);
    
    if (true || (increasing_Q && Q>=Q_max)) {
      
      std::vector<std::vector<int>> partition (M);
    
      for (int y=0; y<basis.size(); y++) {
	int i = basis[y];
	int root = find(i);
	int index = pointer[root];
	partition[index].push_back(y);
      }
      //std::cout <<"hello" << std::endl;
    
      Ps.push_back(interpolationMatrix(N, partition));
      
      //basis = used;
      //N = M;
    
      //increasing_Q = false;
      
      
      //std::vector<int> PI (N+1);
      //for (int i=0; i<N; i++) {
      //	PI[i] = i;
      //}
      //PI[N] = N+1;
      //
      //for (int x=0; x<N; x++) {
      //	int i = basis[i];
      //  find(i);
      //}
      //std::vector<int> PJ (N);
      //for (int x=0; x<N; x++) {
      //	int i = basis[x];
      //	PJ[x] = pointer[find(i)];
      //}
      //std::vector<double> PD (N, 1.0);
      //Ps.push_back(SparseMatrix(PI, PJ, PD, N, M).Transpose());
      
    }   
    
    if (printing) {
      //std::cout << Q_max << std::endl;
      std::cout << "modularity: " << Q << std::endl;
      //std::cout << "Number of iterations: " << num_iter << std::endl;
      //std::cout << "Number of levels: " << Ps.size() + 1 << std::endl;
      std::cout << "level 0: " << A.Rows() << " aggregates" << std::endl;
      for (int i=0; i<Ps.size(); i++) {
	std::cout << "level " << i+1 << ": " << Ps[i].Rows() << " aggregates" << std::endl;
      }
    }
  
    return Ps;
      
  }




}
