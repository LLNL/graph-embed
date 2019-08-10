
// Copyright 2019 Lawrence Livermore National Security. 
// Produced at the Lawrence Livermore National Laboratory.
// LLNL-CODE-781679. All Rights reserved. See file LICENSE for details.
//
// This file is part of graph-embed. For more information and source code
// availability, see github.com/LLNL/graph-embed
//
// SPDX-License-Identifier: LGPL-2.1


#ifndef EMBED_HPP
#define EMBED_HPP

#include "partitioner.hpp"

namespace partition {

  /*
    Embeds A in [0,1]^d via minimization, starting optionally from the given coordinates
   */
  std::vector<std::vector<double>> embedViaMinimization (const SparseMatrix& A, const int d);
  void embedViaMinimization (const SparseMatrix& A, const int d, std::vector<std::vector<double>>& coords, const int ITER=10);

  /*
    Takes in an embedding algorithm and produces the multilevel-algorithm building block.
  */
  std::function<void (const SparseMatrix&, 
		      const SparseMatrix&,
		      const std::vector<int>&,
		      const std::vector<std::vector<double>>&,
		      const std::vector<double>&,
		      std::vector<std::vector<double>>&,
		      const int)>
  anyToMultilevel (std::function<std::vector<std::vector<double>> (const SparseMatrix&, const int d)> embeddingAlgorithm);
  
  /*
    Takes in a hierarhcy, a dimension, and an embedding algorithm and produces coordinates in a multilevel fashion.
   */
  std::vector<std::vector<double>> embedVia(const std::vector<SparseMatrix>& As,
					    const std::vector<SparseMatrix>& P_Ts,
					    const int d,
					    std::function<void (const SparseMatrix&, 
								const SparseMatrix&,
								const std::vector<int>&,
								const std::vector<std::vector<double>>&,
								const std::vector<double>&,
								std::vector<std::vector<double>>&,
								const int)> embedder);
  /*
    Helper for embedVia
  */
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
									   const int)> embedder);

  /*
    Takes in a hierarchy and a dimension and produces coordinates in a multilevel fashion.
   */
  std::vector<std::vector<double>> embed (const std::vector<SparseMatrix>& As, 
					  const std::vector<SparseMatrix>& ps, 
					  const int d);
  std::vector<std::vector<double>> embedMultilevel (const std::vector<SparseMatrix>& As, 
						    const std::vector<SparseMatrix>& ps, 
						    const int d, 
						    const int index, 
						    std::vector<double>& r_A, 
						    std::vector<std::vector<double>>& coords_A);
}

#endif // EMBED_HPP
