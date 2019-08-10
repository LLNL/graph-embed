
// Copyright 2019 Lawrence Livermore National Security. 
// Produced at the Lawrence Livermore National Laboratory.
// LLNL-CODE-781679. All Rights reserved. See file LICENSE for details.
//
// This file is part of graph-embed. For more information and source code
// availability, see github.com/LLNL/graph-embed
//
// SPDX-License-Identifier: LGPL-2.1


#ifndef MATRIXUTILS_HPP
#define MATRIXUTILS_HPP

#include "sparsematrix.hpp"

using SparseMatrix = linalgcpp::SparseMatrix<double>;
using coord = std::vector<double>;
using coordinates = std::vector<coord>;

namespace partition {

  /* 
     @param n the desired dimension
     Returns the n*n identity matrix
  */
  SparseMatrix identity (int n);  
  
  /* 
     @param A the adjacency matrix  
     Given an adjacency matrix A, returns the corresponding graph Laplacian matrix L
  */
  SparseMatrix toLaplacian (const SparseMatrix& A);
  
  /* 
     @param L the graph Laplacian matrix
     Given a graph Laplacian matrix L, returns the corresponding adjacency matrix A
  */
  SparseMatrix fromLaplacian (const SparseMatrix& L); 

}

#endif // MATRIXUTILS_HPP
