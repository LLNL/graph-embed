
// Copyright 2019 Lawrence Livermore National Security. 
// Produced at the Lawrence Livermore National Laboratory.
// LLNL-CODE-781679. All Rights reserved. See file LICENSE for details.
//
// This file is part of graph-embed. For more information and source code
// availability, see github.com/LLNL/graph-embed
//
// SPDX-License-Identifier: LGPL-2.1


#include <cstdlib>
#include <iostream>
#include <ctime>
#include "linalgcpp.hpp"
#include "parser.hpp"
#include "export.hpp"
#include "embed.hpp"

#include <omp.h>

#include <iomanip>
#include <fstream>

#include "partitioner.hpp"

using SparseMatrix = linalgcpp::SparseMatrix<double>;


std::string adjlist = "adjlist";
std::string coolist = "coolist";
std::string table = "table";
std::string csr = "csr";
std::string mtx = "mtx";


  // embed -f [input path] -format [adjlist coolist table csr mtx] -o [output path] [options...]

int main (int argc, char* argv[]) {

  std::string inputpath;
  std::string outputpath;
  std::string format;
  int dimension = 3;
  bool symmetric = false;

  for (int i=0; i<argc-1; i++) {
    if (std::string(argv[i]) == "-f") {
      inputpath = std::string(argv[i+1]);
    } else if (std::string(argv[i]) == "-o") {
      outputpath = std::string(argv[i+1]);
    } else if (std::string(argv[i]) == "-format") {
      format = std::string(argv[i+1]);      
      if (format != adjlist && 
	  format != coolist && 
	  format != table && 
	  format != csr && 
	  format != mtx) {
	std::cerr << "-format must be from [adjlist coolist table csr mtx]" << std::endl;
	return 1;
      }
    } else if (std::string(argv[i]) == "-dimension") {
      dimension = std::stoi(std::string(argv[i+1]));
    } else if (std::string(argv[i]) == "-symmetric") {
      symmetric = std::string(argv[i+1]) == "true";
    }
  }

  if (inputpath == "") {
    std::cerr << "-f must be present with one argument" << std::endl;
    return 1;
  } else if (outputpath == "") {
    std::cerr << "-o must be present with one argument" << std::endl;
    return 1;
  } else if (format == "") {
    std::cerr << "--inputformat must be present with one argument" << std::endl;
    return 1;
  }

  SparseMatrix A;
  if (format == adjlist) {
    A = linalgcpp::ReadAdjList(inputpath, symmetric);
  } else if (format == coolist) {
    A = linalgcpp::ReadCooList(inputpath, symmetric); 
  } else if (format == table) {
    A = linalgcpp::ReadTable<double>(inputpath);
  } else if (format == csr) {
    A = linalgcpp::ReadCSR(inputpath);
  } else if (format == mtx) {
    A = linalgcpp::ReadMTX(inputpath);
  }

  double coarseningFactor = 0.1;
  std::vector<SparseMatrix> hierarchy = partition::partition(A, coarseningFactor);
  std::vector<SparseMatrix> As = {A};
  for (int level=0; level<k; level++) {
    As.push_back(hierarchy[level].Mult(As[level]).Mult(hierarchy[level].Transpose()));
  }
  
  std::vector<std::vector<double>> coords = partition::embed(As, hierarchy, dimension);

  partition::writeCoords(coords, outputpath);

}
