
// Copyright 2019 Lawrence Livermore National Security. 
// Produced at the Lawrence Livermore National Laboratory.
// LLNL-CODE-781679. All Rights reserved. See file LICENSE for details.
//
// This file is part of graph-embed. For more information and source code
// availability, see github.com/LLNL/graph-embed
//
// SPDX-License-Identifier: LGPL-2.1


#include "export.hpp"

namespace partition {
  
  void writePartition (const std::vector<int>& partition, const std::string& outputpath) {
    std::ofstream file;
    file.open(outputpath);

    for (int i=0; i<partition.size(); i++) {
      file << partition[i] << "\n";
    }
    
    file.close();
  }

  void writeCoords (const std::vector<std::vector<double>>& coords, const std::string& outputpath) {
    std::ofstream file;
    file.open(outputpath);
    
    for (int i=0; i<coords.size(); i++) {
      for (int j=0; j<coords[i].size(); j++) {
	file << coords[i][j] << " ";
      }
      file << "\n";
    }

    file.close();
  }
}
