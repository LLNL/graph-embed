
// Copyright 2019 Lawrence Livermore National Security. 
// Produced at the Lawrence Livermore National Laboratory.
// LLNL-CODE-781679. All Rights reserved. See file LICENSE for details.
//
// This file is part of graph-embed. For more information and source code
// availability, see github.com/LLNL/graph-embed
//
// SPDX-License-Identifier: LGPL-2.1


#ifndef  FORCEATLAS_HPP
#define  FORCEATLAS_HPP

#include <omp.h>

#include <vector>

#include "matrixutils.hpp"

/* iterations is the number of iterations to be performed
   linlog is a variant which uses logarithmic attraction force F <- log (1+F)
   pos is the table (NumberOfNodes x dimension) of the initial locations of points, if specified
   nohubs is a variant in which nodes with high indegree have more central position than nodes with outdegree (for directed graphs) 
   k is the repel constant : the greater the constant k the stronger the repulse force between points 
   gravity is the gravity constant : indicates how strongly the nodes should be attracted to the center of gravity
   ks is the speed constant : the greater the value of ks the more movement the nodes make under the acting forces
   ksmax limits the speed from above
   delta is the parameter to modify attraction force; means that weights are raised to the power = delta
   center is the center of gravity  
   tolerance is the tolerance to swinging constant
   dim is the dimension
*/

/* Forces_prev_k_i is the previous forces at i for k
   Fr_ij = RepulsionForce_ij = (d_i+1) * (d_j+1) * k / distance_ij
   Fa_ij = AttractionForce_ij = fa, where fa = distance(i,j); if linlog fa = log(1+fa); fa = a_ij^delta * fa; if nohubs fa = fa/(d_i + 1)
   should probably just use: fa = a_ij^delta * log(1 + distance(i,j))
   mylist_k_ij = (pos_i_k - pos_j_k) / distance(i,j)
   Far_k_i = Sum_j [mylist_k_ij * (RepulsionForce_ij - AttractionForce_ij)]
   asumming the center is 0,
   rowSum_i = Sum_k [pos_i_k]
   uv2_k_i = -pos_i_k/rowSum_i
   Fg_k_i = GravityForce_k_i = uv2_k_i * gravity * (d_i + 1)
   Forces_k_i = Far_k_i + Fg_k_i
   swing_i = abs(sqrt(Sum_k [(Forces_k_i - Forces_prev_k_i)^2]))
   globalSwing = Sum_i [(d_i+1) * swing_i]
   traction_i = abs(sqrt(Sum_k [(Forces_k_i + Forces_prev_k_i)^2]))/2
   globalTraction = Sum_i [(d_i + 1) * traction_i]
   globalSpeed = tolerate * globalTraction / globalSwing
   speed_i = ks * globalSpeed / (1 + globalSpeed * sqrt(swing_i))
   totalF_i = sqrt(Sum_k [Forces_k_i ^ 2])
   speedConstraint_i = ksmax / totalF_i
   speed_i = (speed_i > speedConstraint_i) : speedConstraint_i ? speed_i
   displacement_k_i = Forces_k_i + speed_i
   position_i_k = displacement_k_i + position_i_k
   Forces_prev = Forces
*/

// https://github.com/adolfoalvarez/Force-Atlas-2/blob/master/ForceAtlas2.R
// http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0098679


namespace partition {
  
  inline double abs (double val) {
    return  (val < 0) ? -val : val;
  }
  
  inline double distance (const std::vector<double>& v1, const std::vector<double>& v2) {
    assert(v1.size() == v2.size());
    double sum = 0.0;
    for (int i=0; i<v1.size(); i++) {
      double d = v2[i] - v1[i];
      sum += d * d;
    }
    return sqrt(sum);
  }

  inline double magnitude (const std::vector<double>& v) {
    double sum = 0.0;
    for (int i=0; i<v.size(); i++) {
      double d = v[i];
      sum += d * d;
    }
    return sqrt(sum);
  }
  
  void forceAtlas (const SparseMatrix& A,
		   const int dim,
		   std::vector<std::vector<double>>& coords,
		   const int iterations=100000,
		   const double ks=0.1,
		   const double ksmax=1.0,
		   const double repel=1.0,
		   const double attract=1.0,
		   const double gravity=1.0,
		   const bool useWeights=true,
		   const bool linlog=false,
		   const bool nohubs=false, 
		   const double delta=1.0,
		   const double tolerate=1.0,
		   const bool normalize=false) {
    std::random_device rd;
    std::mt19937 gen (rd());
    //unsigned seed = 0;
    //std::default_random_engine gen (seed);
    std::uniform_real_distribution<double> random (-1.0, 1.0);

    double epsilon = 0.00001;

    auto& I = A.GetIndptr();
    auto& J = A.GetIndices();
    auto& D = A.GetData();
    
    int n = A.Rows();
    
    if (coords.size() == 0) {
      coords = std::vector<std::vector<double>> (n, std::vector<double> (dim));
      for (int i=0; i<n; i++) {
	for (int k=0; k<dim; k++) {
	  coords[i][k] = random(gen);
	}
      }
    }
    
    std::vector<double> deg (n);
    if (useWeights) {
      for (int i=0; i<n; i++) {
	double sum = 0.0;
	for (int k=I[i]; k<I[i+1]; k++) {
	  sum += D[k]; 
	}
	deg[i] = sum;
      }
    } else {
      for (int i=0; i<n; i++) {
	deg[i] = 1.0 * (I[i+1] - I[i]);
      }
    }
    
    std::vector<std::vector<double>> forces_prev (n, std::vector<double> (dim));
    std::vector<std::vector<double>> forces (n, std::vector<double> (dim));
    std::vector<double> swing (n);

    for (int iter=0; iter<iterations; iter++) {
      #pragma omp parallel for
      for (int i=0; i<n; i++) {
	std::vector<double> force_i (dim, 0.0);
	double deg_ip1 = deg[i] + 1;
	for (int j=0; j<n; j++) {
	  if (i != j) {
	    double deg_jp1 = deg[j] + 1;
	    double dis_ij = distance (coords[i], coords[j]);
	    if (dis_ij < epsilon) {
	      dis_ij = epsilon;
	    }
	    double val = deg_ip1 * deg_jp1 * repel / (dis_ij * dis_ij);
	    double Fr_ij = val;
	    
	    for (int k=0; k<dim; k++) {
	      double direction = - (coords[j][k] - coords[i][k]) / dis_ij;
	      double Fr_sum = direction * Fr_ij;
	      force_i[k] += Fr_sum;
	    }
	  }
	}
	
	for (int k2=I[i]; k2<I[i+1]; k2++) {
	  int j = J[k2];
	  double dis_ij = distance (coords[i], coords[j]);
	  if (dis_ij < epsilon) {
	    dis_ij = epsilon;
	  }
	  
	  double fa_ij = dis_ij;
	  if (linlog) {
	    fa_ij = log(1+fa_ij);
	  }
	  double a_ij;
	  if (useWeights) {
	    a_ij = D[k2];
	  } else {
	    a_ij = 1.0;
	  }
	  
	  if (delta == 1.0) {
	    fa_ij = fa_ij * a_ij;
	  } else if (delta != 0.0) {
	    fa_ij = (a_ij < 0 ? -1 : 1) * pow(abs(a_ij), delta) * fa_ij;
	  }
	  
	  if (nohubs) {
	    fa_ij = fa_ij / deg_ip1;
	  }
	  double Fa_ij = attract * fa_ij;
	  
	  for (int k=0; k<dim; k++) {
	    double direction = (coords[j][k] - coords[i][k]) / dis_ij;	    
	    double Fa_sum = direction * Fa_ij;
	    force_i[k] += Fa_sum;
	  }
	}
	
	double mag = magnitude(coords[i]);
	for (int k=0; k<dim; k++) {
	  double Far_ki = force_i[k];
	  double uv2_ki = -coords[i][k] / mag;
	  double Fg_ki = uv2_ki * gravity * deg_ip1;
	  forces[i][k] = Far_ki + Fg_ki;
	}
      }

      #pragma omp parallel for
      for (int i=0; i<n; i++) {
	swing[i] = distance(forces[i], forces_prev[i]);
      }
      
      double globalSwing = 0.0;
      
      for (int i=0; i<n; i++) {
	globalSwing += (deg[i]+1) * swing[i];
      }
      if (globalSwing < epsilon) {
	globalSwing = epsilon;
      }
      
      globalSwing = 1.0;
      
      double globalTraction = 0.0;
      
      for (int i=0; i<n; i++) {
	double sum = 0.0;
	for (int k=0; k<dim; k++) {
	  double val = forces[i][k] + forces_prev[i][k];
	  sum += val * val;
	}
	double traction_i = sqrt(sum)/2;
	globalTraction += (deg[i]+1) * traction_i;
      }
      
      globalTraction = 1.0;
      
      double globalSpeed = tolerate * globalTraction / globalSwing;
      
      #pragma omp parallel for
      for (int i=0; i<n; i++) {
	double totalF_i = magnitude(forces[i]);
	
	double speed_i = ks * globalSpeed / (1 + globalSpeed * sqrt(swing[i]));
	double speedConstraint_i = ksmax / totalF_i; 
	
	if (speed_i > speedConstraint_i) {
	  speed_i = speedConstraint_i;
	}
	
	for (int k=0; k<dim; k++) {
	  double displacement_ik = forces[i][k] * speed_i;
	  coords[i][k] = displacement_ik + coords[i][k];
	}
      }

      forces_prev = forces;
      #pragma omp parallel for
      for (int i=0; i<n; i++) {
	for (int k=0; k<dim; k++) {
	  forces[i][k] = 0.0;
	}
      }
    }

    if (normalize) {
      // center coords at 0
      std::vector<double> avg (dim);
      for (int i=0; i<n; i++) {
	for (int k=0; k<dim; k++) {
	  avg[k] = avg[k] + coords[i][k];
	}
      }
      for (int k=0; k<dim; k++) {
	avg[k] = avg[k] / n;
      }
      #pragma omp parallel for
      for (int i=0; i<n; i++) {
	for (int k=0; k<dim; k++) {
	  coords[i][k] -= avg[k];
	}
      }

      double max_length = 0.0;
      for (int i=0; i<n; i++) {
	double length = magnitude(coords[i]);
	if (max_length < length) {
	  max_length = length;
	}
      }
      #pragma omp parallel for
      for (int i=0; i<n; i++) {
	for (int k=0; k<dim; k++) {
	  coords[i][k] = coords[i][k] / max_length;
	}
      }
    }
    return;
  }

  std::vector<std::vector<double>> forceAtlas (const SparseMatrix& A,
					       const int dim=2) {
    std::vector<std::vector<double>> coords (0);
    forceAtlas(A, dim, coords);
    return coords;
  }

  void forceAtlasMultilevel (const SparseMatrix& A,
			     const SparseMatrix& P,
			     const std::vector<int>& v_A,
			     const std::vector<std::vector<double>>& coords_A,
			     const std::vector<double>& r_A,
			     std::vector<std::vector<double>>& coords,
			     int dim=2,
			     int iterations=10,
			     double ks=0.1,
			     double ksmax=1.0,
			     bool useWeights=true,
			     bool linlog=false,
			     bool nohubs=false, 
			     double repel=1.0,
			     double attract=1.0,
			     double gravity=1.0,
			     double delta=1.0,
			     double tolerate=1.0) {
    std::random_device rd;
    std::mt19937 gen (rd());
    //unsigned seed = 0;
    //std::default_random_engine gen (seed);
    std::uniform_real_distribution<double> random (-1.0, 1.0);
    double epsilon = 0.00001;
    int m = P.Rows();

    #pragma omp parallel for
    for (int a=0; a<m; a++) {
      auto& I = A.GetIndptr();
      auto& J = A.GetIndices();
      auto& D = A.GetData();
      auto& PI = P.GetIndptr();
      auto& PJ = P.GetIndices();
      std::vector<int> v (PI[a+1] - PI[a]);
      int index = 0;
      for (int c=PI[a]; c<PI[a+1]; c++) {
	v[index] = PJ[c];
	index++;
      }

      int n = v.size();

      for (int i=0; i<n; i++) {
	for (int k=0; k<dim; k++) {
	  coords[v[i]][k] = random(gen); //coords_A[a][k] + random(gen);
	}
      }
    
      std::vector<double> deg (n);
      if (useWeights) {
	for (int i=0; i<n; i++) {
	  double sum = 0.0;
	  for (int k=I[v[i]]; k<I[v[i]+1]; k++) {
	    if (v_A[J[k]] == a) {
	      sum += D[k];
	    }
	  }
	  deg[i] = sum;
	}
      } else {
	for (int i=0; i<n; i++) {
	  double sum = 0.0;
	  for (int k=I[v[i]]; k<I[v[i]+1]; k++) {
	    if (v_A[J[k]] == a) {
	      sum += 1.0;
	    }
	  }
	  deg[i] = sum;
	}
      }
    
      std::vector<std::vector<double>> forces_prev (n, std::vector<double> (dim, 0.0));
      std::vector<double> Fr_sums (n, 0.0);
      std::vector<std::vector<double>> forces (n, std::vector<double> (dim, 0.0));
      std::vector<double> swing (n, 0.0);
    
      for (int iter=0; iter<iterations; iter++) {
	for (int i=0; i<n; i++) {
	  std::vector<double> force_i (dim, 0.0);
	  double deg_ip1 = deg[i] + 1;
	  for (int j=0; j<n; j++) {
	    if (i != j) {
	      double deg_jp1 = deg[j] + 1;
	      double dis_ij = distance (coords[v[i]], coords[v[j]]);
	      if (dis_ij < epsilon) {
		dis_ij = epsilon;
	      }
	      double val = deg_ip1 * deg_jp1 * repel / (dis_ij * dis_ij);
	      double Fr_ij = val;
	    
	      for (int k=0; k<dim; k++) {
		double direction = - (coords[v[j]][k] - coords[v[i]][k]) / dis_ij;
		double Fr_sum = direction * Fr_ij;
		force_i[k] += Fr_sum;
	      }
	    }
	  }
	  double mag = magnitude(coords[v[i]]);
	  if (mag < epsilon) {
	    mag = epsilon;
	  }
	  for (int k2=I[v[i]]; k2<I[v[i]+1]; k2++) {
	    int j = J[k2];
	    if (v_A[j] == a && j != i) {
	      // internal
	      double dis_ij = distance (coords[v[i]], coords[j]);
	      if (dis_ij < epsilon) {
		dis_ij = epsilon;
	      }
	      
	      double fa_ij = dis_ij;
	      if (linlog) {
		fa_ij = log(1+fa_ij);
	      }
	      double a_ij;
	      if (useWeights) {
		a_ij = D[k2];
	      } else {
		a_ij = 1.0;
	      }
	      
	      if (delta == 1.0) {
		fa_ij = fa_ij * a_ij;
	      } else if (delta != 0.0) {
		fa_ij = (a_ij < 0 ? -1 : 1) * pow(abs(a_ij), delta) * fa_ij;
	      }
	      
	      if (nohubs) {
		fa_ij = fa_ij / deg_ip1;
	      }
	      double Fa_ij = attract * fa_ij;
	      
	      for (int k=0; k<dim; k++) {
		double direction = (coords[j][k] - coords[v[i]][k]) / dis_ij;	    
		double Fa_sum = direction * Fa_ij;
		force_i[k] += Fa_sum;
	      }
	    } else {
	      // external
	      double pull = 100.0;
	      double dis_ij = distance (coords_A[a], coords_A[v_A[j]]);
	      if (dis_ij < epsilon) {
		dis_ij = epsilon;
	      }
	      double fao_ij = 1.0; // dis_ij;
	      double Fao_ij = pull * fao_ij;
	      
	      for (int k=0; k<dim; k++) {
		double direction = (coords_A[v_A[j]][k] - coords_A[a][k]) / dis_ij;
		double Fao_sum = direction * Fao_ij / mag;
		force_i[k] += Fao_sum;
	      }
	    }
	  }
	  
	  for (int k=0; k<dim; k++) {
	    double Far_ki = force_i[k]; // = Fr_sum + Fa_sum + Fao_sum;
	    double uv2_ki = -coords[v[i]][k] / mag; 
	    double Fg_ki = uv2_ki * gravity * deg_ip1;
	    forces[i][k] = Far_ki + Fg_ki;
	  }
	}

	for (int i=0; i<n; i++) {
	  double sum = 0.0;
	  for (int k=0; k<dim; k++) {
	    double val = forces[i][k] - forces_prev[i][k];
	    sum += val * val;
	  }
	  swing[i] = sqrt(sum);
	  if (swing[i] < epsilon) {
	    swing[i] = epsilon;
	  }
	}
      
	double globalSwing = 0.0;
      
	for (int i=0; i<n; i++) {
	  globalSwing += (deg[i]+1) * swing[i];
	}
	if (globalSwing < epsilon) {
	  globalSwing = epsilon;
	}
      
	globalSwing = 1.0;
      
	double globalTraction = 0.0;
      
	for (int i=0; i<n; i++) {
	  double sum = 0.0;
	  for (int k=0; k<dim; k++) {
	    double val = forces[i][k] + forces_prev[i][k];
	    sum += val * val;
	  }
	  double traction_i = sqrt(sum)/2;
	  globalTraction += (deg[i]+1) * traction_i;
	}
      
	globalTraction = 1.0;
      
	double globalSpeed = tolerate * globalTraction / globalSwing;
      
	for (int i=0; i<n; i++) {
	  double totalF_i = magnitude(forces[i]);
	
	  double speed_i = ks * globalSpeed / (1 + globalSpeed * sqrt(swing[i]));
	  double speedConstraint_i = ksmax / totalF_i; 
	
	  if (speed_i > speedConstraint_i) {
	    speed_i = speedConstraint_i;
	  }
	
	  for (int k=0; k<dim; k++) {
	    double displacement_ik = forces[i][k] * speed_i;
	    coords[v[i]][k] = displacement_ik + coords[v[i]][k];
	  }
	}
      
	forces_prev = forces;
	for (int i=0; i<n; i++) {
	  for (int k=0; k<dim; k++) {
	    forces[i][k] = 0.0;
	  }
	}
      }
      {
	std::vector<double> avg (dim);
	for (int i=0; i<n; i++) {
	  for (int k=0; k<dim; k++) {
	    avg[k] = avg[k] + coords[v[i]][k];
	  }
	}
	for (int k=0; k<dim; k++) {
	  avg[k] = avg[k] / n;
	}
	for (int i=0; i<n; i++) {
	  for (int k=0; k<dim; k++) {
	    coords[v[i]][k] -= avg[k];
	  }
	}

	double max = 0.0;
	for (int i=0; i<n; i++) {
	  double sum = magnitude(coords[v[i]]);
	  if (sum > max) {
	    max = sum;
	  }
	}
	if (max < epsilon) {
	  max = epsilon;
	}
	for (int i=0; i<n; i++) {
	  for (int k=0; k<dim; k++) {
	    coords[v[i]][k] = coords_A[a][k] + r_A[a] * (coords[v[i]][k] / max);
	  }
	}
      }

    }
    return;
  }


} // namespace partition

#endif // FORCEATLAS_HPP
