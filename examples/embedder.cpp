
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
#include <algorithm>

#include "linalgcpp.hpp"
#include "parser.hpp"
#include "export.hpp"
#include "embed.hpp"

#include <iomanip>
#include <fstream>

#include "partitioner.hpp"

using SparseMatrix = linalgcpp::SparseMatrix<double>;

std::string adjlist = "adjlist";
std::string coolist = "coolist";
std::string table = "table";
std::string csr = "csr";
std::string mtx = "mtx";

SparseMatrix largestComponent (const SparseMatrix& A) {
  const std::vector<int>& I = A.GetIndptr();
  const std::vector<int>& J = A.GetIndices();
  int n = A.Rows();
  std::vector<int> components (n, -1);
  int component = 0;
  int prev = 0;
  while (prev < n) {
    if (components[prev] == -1) {
      std::vector<int> stack;
      std::vector<int> off;
      components[prev] = component;
      stack.push_back(prev);
      off.push_back(0);
      while (!stack.empty()) {
	int curr = stack.back();
	int offset = off.back();
	bool found = false;
	for (int k=offset; k<I[curr+1] - I[curr]; k++) {
	  int next=J[k + I[curr]];
	  if (components[next] == -1) {
	    components[next] = component;
	    off.pop_back();
	    off.push_back(k+1);
	    stack.push_back(next);
	    off.push_back(0);
	    found = true;
	    break;
	  }
	}
	if (!found) {
	  stack.pop_back();
	  off.pop_back();
	}
      }
      component++;
    }
    prev++;
  }
  std::cout << "num components: " << component << std::endl;
  std::vector<int> count (component, 0);
  for (int i=0; i<n; i++) {
    count[components[i]]++;
  }
  int max = -1, maxIndex = -1;
  for (int i=0; i<component; i++) {
    if (count[i] > max) {
      max = count[i];
      maxIndex = i;
    }
  }
  std::vector<int> vertices;
  for (int i=0; i<n; i++) {
    if (components[i] == maxIndex) {
      vertices.push_back(i);
    }
  }
  return A.GetSubMatrix(vertices, vertices);
}

SparseMatrix removeLargest (const SparseMatrix& A) {
  const std::vector<int>& I = A.GetIndptr();
  const std::vector<int>& J = A.GetIndices();
  int n = A.Rows();

  int max = 0;
  for (int i=0; i<n; i++) {
    if (I[i+1] - I[i] > max) {
      max = I[i+1] - I[i];
    }
  }
  std::vector<int> vertices;
  for (int i=0; i<n; i++) {
    if (I[i+1] - I[i] < 0.01 * max) {
      vertices.push_back(i);
    }
  }
  return A.GetSubMatrix(vertices, vertices);
  
}

void printMatrix(SparseMatrix& A) {
  int n = A.Rows();
  int m = A.Cols();
  auto I = A.GetIndptr();
  auto J = A.GetIndices();
  for (int i=0; i<n; i++) {
    int cur_j = 0;
    for (int k=I[i]; k<I[i+1]; k++) {
      int j = J[k];
      while (cur_j < j) {
	std::cout << "- ";
	cur_j++;
      }
      std::cout << "1 ";
      cur_j = j+1;
    }
    while (cur_j < m) {
	std::cout << "- ";
	cur_j++;
    }
    std::cout << std::endl;
  }
}

int main (int argc, char* argv[]) {
  std::vector<std::string> inputpaths = {
    // your graphs here
  };

  for (int x=0; x<inputpaths.size(); x++) {
    std::string inputpath = inputpaths[x];
    std::cout << "doing: " << inputpath << std::endl;

    SparseMatrix A;

    {
      A = linalgcpp::ReadAdjList(inputpath, true);
      bool connected = true;
      if (connected) {
	std::cout << "before: " << A.Rows() << std::endl;
	A = largestComponent(A);
	std::cout << "after: " << A.Rows() << std::endl;
      }

      bool random = false;
      if (random) {
	std::random_device rd;
	std::mt19937 gen (rd());
	//unsigned seed = 0;
	//std::default_random_engine gen (seed);
    
	double epsilon = 0.001;
	std::uniform_real_distribution<double> random(1-epsilon, 1+epsilon);
    
	std::vector<double> vertex_weights (A.Rows());
    
	for (int i=0; i<A.Rows(); i++) {
	  vertex_weights[i] = random(gen);
	}
    
	A.ScaleRows(vertex_weights);
	A.ScaleCols(vertex_weights);
      }
    }

    std::cout << A.Rows() << " " << A.Cols() << " " << A.GetData().size() << std::endl;
    int n = A.Rows();

    std::cout << "input read" << std::endl;

    double coarseningFactor = 1.0 / 10.0; 
    std::vector<SparseMatrix> hierarchy = partition::partition(A, coarseningFactor, false, true, 1.0, 1, false);

    int killNum = 0;
    for (int i=0; i<killNum; i++) {
      hierarchy.pop_back();
    }

    std::cout << "partitioned!" << std::endl;

    int startLevel = 0;
    for (int i=0; i<startLevel; i++) {
      A = hierarchy[0].Mult(A).Mult(hierarchy[0].Transpose());
      hierarchy.erase(hierarchy.begin());
    }
    n = A.Rows();

    int k = hierarchy.size();
    std::cout << A.Rows() << " ";
    for (int i=0; i<hierarchy.size(); i++) {
      std::cout << hierarchy[i].Rows() << " ";
    }
    std::cout << std::endl;

    k = hierarchy.size();

    int dimension = 3;
    std::vector<SparseMatrix> As = {A};
    for (int level=0; level<k; level++) {
      As.push_back(hierarchy[level].Mult(As[level]).Mult(hierarchy[level].Transpose()));
    }

    std::cout << "starting embedding: " << std::endl;
    linalgcpp::Timer timer (linalgcpp::Timer::Start::True);
    std::vector<std::vector<double>> coords = partition::embed(As, hierarchy, dimension);
    timer.Click();
    std::cout << "embedded! in time " << timer[0] << "s" << std::endl;

    for (int i=0; i<coords.size(); i++) {
      for (int k=0; k<dimension; k++) {
	assert(!isnan(coords[i][k]));
      }
    }

    std::string partpath = "temp/part.temp";
    std::string coordspath = "temp/coords.temp";
    std::string plotpath = "temp/plot.html";
    std::string matpath = "temp/mat.temp";

    // write partition
    std::ofstream partfile;
    partfile.open(partpath);
    if (k == 0) {
      std::vector<std::vector<int>> level (n);
      for (int i=0; i<n; i++) {
	level[i].push_back(i);
      }
      k = 1;
      hierarchy = {partition::interpolationMatrix(n, level)};
    }
    partfile << n << " " << k << "\n";
    for (int i=0; i<k; i++) {
      partfile << hierarchy[i].Rows() << " ";
    }
    partfile << "\n";
    for (int level=0; level<k; level++) {
      SparseMatrix& P = hierarchy[level];
      auto& I = P.GetIndptr();
      auto& J = P.GetIndices();
      for (int i=0; i<P.Rows(); i++) {
	for (int k=I[i]; k<I[i+1]; k++) {
	  int j = J[k];
	  partfile << j << " ";
	}
	partfile << "\n";
      }
    }
    partfile.close();

    // write coords
    std::ofstream coordsfile;
    coordsfile.open(coordspath);
    for (int i=0; i<n; i++) {
      if (dimension == 2) {
	coordsfile << coords[i][0] << " " << coords[i][1] << " " << 0.0 << "\n";
      }
      if (dimension == 3) {
	coordsfile << coords[i][0] << " " << coords[i][1] << " " << coords[i][2] << "\n";
      }
    }
    coordsfile.close();

    // write matrix:
    std::ofstream matfile;
    matfile.open(matpath);
    std::vector<int>& I = A.GetIndptr();
    std::vector<int>& J = A.GetIndices();
    for (int i=0; i<A.Rows(); i++) {
      for (int k=I[i]; k<I[i+1]; k++) {
	int j = J[k];
	matfile << i << " " << j << "\n";
      }
    }
    matfile.close();
  
    std::string call = "python scripts/plot-graph.py -graph " + matpath + " -part " + partpath + " -coords " + coordspath + " -o " + plotpath;
    std::cout << call << std::endl;
    system(call.c_str());
  }
}










