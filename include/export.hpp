
// Copyright 2019 Lawrence Livermore National Security. 
// Produced at the Lawrence Livermore National Laboratory.
// LLNL-CODE-781679. All Rights reserved. See file LICENSE for details.
//
// This file is part of graph-embed. For more information and source code
// availability, see github.com/LLNL/graph-embed
//
// SPDX-License-Identifier: LGPL-2.1


#ifndef EXPORT_HPP
#define EXPORT_HPP

#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include "matrixutils.hpp"
  
namespace partition {
  void writePartition (const std::vector<int>&, const std::string& outputpath);

  void writeCoords (const std::vector<std::vector<double>>& coords, const std::string& outputpath);

}

#endif // EXPORT_HPP
