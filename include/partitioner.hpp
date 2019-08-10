
// Copyright 2019 Lawrence Livermore National Security. 
// Produced at the Lawrence Livermore National Laboratory.
// LLNL-CODE-781679. All Rights reserved. See file LICENSE for details.
//
// This file is part of graph-embed. For more information and source code
// availability, see github.com/LLNL/graph-embed
//
// SPDX-License-Identifier: LGPL-2.1


#ifndef PARTITIONER_HPP
#define PARTITIONER_HPP

#include "matrixutils.hpp"
#include "linalgcpp.hpp"

using CooMatrix = linalgcpp::CooMatrix<double>;

namespace partition {

  /*
    Creates P^T from partition such that P^TAP is the coarse version of A
  */ 
  SparseMatrix interpolationMatrix (const int numCols, const std::vector<std::vector<int>>& partition);

  /*
    Computes the modularity of the adjacency matrix A under P^T where P^TAP is the coarse A
   */
  double modularity (const SparseMatrix& A, const SparseMatrix& P_T);


  SparseMatrix partitionTest (const SparseMatrix& A, const float stallStopThreshold=1.0);
  SparseMatrix partitionBase (const SparseMatrix& A, const float stallStopThreshold=1.0);
  SparseMatrix partitionBase2 (const SparseMatrix& A, const float stallStopThreshold=1.0);
  
  /*
    Returns a partition P^T of the adjacency matrix A with as high of modularity as it finds such that P^TAP gives the coarse matrix
   */
  SparseMatrix partition (const SparseMatrix& A, 
			  const bool printing=false, const bool positiveMerging=true, const double stallStopThreshold=1.0, const int matchingIterations=2, const bool mergeLeaves=false);

  /*
    Returns a partition P^T of the adjacency matrix A with as high of modularity as it finds with about numParts sets such that P^TAP gives the coarse matrix
   */
  SparseMatrix partition (const SparseMatrix& A, const int numParts, 
			  const bool printing=false, const bool positiveMerging=true, const double stallStopThreshold=1.0, const int matchingIterations=2, const bool mergeLeaves=false);

  /*
    Returns a hierarchy of partitions of the adjacency matrix A with as high of modularity as it finds such that the coarsening factors between the layers is coarseningFactor
   */
  std::vector<SparseMatrix> partition (const SparseMatrix& A, const double coarseningFactor, 
				       const bool printing=false, const bool positiveMerging=true, const double stallStopThreshold=1.0, const int matchingIterations=2, const bool mergeLeaves=false);

}

#endif // PARTITIONER_HPP
