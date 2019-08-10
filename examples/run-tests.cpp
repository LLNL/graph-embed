
// Copyright 2019 Lawrence Livermore National Security. 
// Produced at the Lawrence Livermore National Laboratory.
// LLNL-CODE-781679. All Rights reserved. See file LICENSE for details.
//
// This file is part of graph-embed. For more information and source code
// availability, see github.com/LLNL/graph-embed
//
// SPDX-License-Identifier: LGPL-2.1


#include <iostream>
#include <iomanip>
#include <fstream>

#include <vector>

#include "linalgcpp.hpp"
#include "partitioner.hpp"
#include "export.hpp"



/* Returns a pair of the vertex aggregte relationship and the aggregate vertex relation ship, while the file has format
   vertex_1,1 vertex_1,2 vertex_1,3 ... vertex_1,n_1
   vertex_2,1 ... vertex_2,n_2
   ...
   vertex_m,1 ... vertex_m,n_m
 */
std::tuple<std::vector<int>, std::vector<std::vector<int>>> readAggregates (std::string communitiesPath) {
  /* TOOD */
  // see: linalgcpp::ReadTable(communitiesPath)

  std::vector<int> vertex_agg (0);
  std::vector<std::vector<int>> agg_vertex (0);
  return std::make_tuple(vertex_agg, agg_vertex);
}


// first part taken from Louvain code
// returns P_T s.t. P_T A P = A^C
SparseMatrix readLouvainAggregates (std::string path) {
  std::vector<std::vector<int> >levels;

  std::ifstream finput;
  finput.open(path,std::fstream::in);

  int l=-1;
  while (!finput.eof()) {
    int node, nodecomm;
    finput >> node >> nodecomm;
    //std::cout << node << " " << nodecomm << std::endl;
    if (finput) {
      if (node==0) {
	l++;
	levels.resize(l+1);
      }
      levels[l].push_back(nodecomm);
    }
  }

  int n = levels[0].size();

  std::vector<int> vertex_agg (n);
  for (int i=0; i<n; i++) {
    vertex_agg[i] = i;
  }
  for (int level=0; level<levels.size(); level++) {
    for (int i=0; i<n; i++) {
      int prev_agg = vertex_agg[i];
      int next_agg = levels[level][prev_agg];
      vertex_agg[i] = next_agg;
    }
  }
  
  int A = 0;
  for (int i=0; i<n; i++) {
    if (A < vertex_agg[i] + 1) {
      A = vertex_agg[i] + 1;
    }
  }

  std::vector<std::vector<int>> agg_vertex (A);

  for (int i=0; i<n; i++) {
    agg_vertex[vertex_agg[i]].push_back(i);
  }
  return partition::interpolationMatrix(n, agg_vertex);
}



// returns P_T such that P_T A P = A'
SparseMatrix readVectorCommunities (std::string communitiesPath) {
  return readLouvainAggregates(communitiesPath);
}

std::tuple<double, double, double, double, double, double> compareComm (std::vector<int>& comm1, std::vector<int>& comm2) {

  assert(comm1.size() == comm2.size());

  int comm1_implies_comm2 = 0;
  int comm2_implies_comm1 = 0;
  int comm1_and_comm2 = 0;

  int total_comm1 = 0;
  int total_comm2 = 0;

  int n = comm1.size();
  for (int i=0; i<n; i++) {
    for (int j=i+1; j<n; j++) {
      if (comm1[i] == comm1[j]) {
	total_comm1++;
      }
      if (comm2[i] == comm2[j]) {
	total_comm2++;
      }
      if ((comm1[i] == comm1[j]) && (comm2[i] == comm2[j])) {
	comm1_and_comm2++;
      }
    }
  }

  return std::make_tuple((1.0 * comm1_implies_comm2) / (1.0 * total_comm1),
			 (1.0 * comm1_implies_comm2) / (1.0 * total_comm2),
			 (1.0 * comm2_implies_comm1) / (1.0 * total_comm1),
			 (1.0 * comm2_implies_comm1) / (1.0 * total_comm2),
			 (1.0 * comm1_and_comm2) / (1.0 * total_comm1),
			 (1.0 * comm1_and_comm2) / (1.0 * total_comm2));

}


const std::string adjlist = "adjlist";
const std::string coolist = "coolist";
const std::string table = "table";
const std::string csr = "csr";
const std::string mtx = "mtx";

const bool doCommunities = true;
const bool doHeuristics = false;
const bool randomizeMatrix = true;
const bool printTimes = true;

void runTest (std::string graphPathPrefix, std::string format, bool hasTrueCommunities) {

  std::string graphPath = graphPathPrefix + ".edges";
  
  SparseMatrix orig;
  if (format == adjlist) {
    orig = linalgcpp::ReadAdjList(graphPath, false); // assume symmetric
  } else if (format == coolist) {
    orig = linalgcpp::ReadCooList(graphPath, false); // assume symmetric
  } else if (format == table) {
    orig = linalgcpp::ReadTable<double>(graphPath);
  } else if (format == csr) {
    orig = linalgcpp::ReadCSR(graphPath);
  } else if (format == mtx) {
    orig = linalgcpp::ReadMTX(graphPath);
  }

  double epsilon = 0.01;
  SparseMatrix A = orig;
  if (randomizeMatrix) {
    //std::random_device rd;
    //std::array<int, std::mt19937::state_size> seed_data;
    //std::generate_n(seed_data.data(), seed_data.size(), std::ref(rd));
    //std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
    //std::mt19937 gen(seq);
    //
    //std::uniform_real_distribution<double> random(1-epsilon, 1+epsilon);
    //
    //std::vector<double> vertex_weights (A.Rows());
    //
    //for (int i=0; i<A.Rows(); i++) {
    //  vertex_weights[i] = random(gen);
    //}
    //
    //A.ScaleRows(vertex_weights);
    //A.ScaleCols(vertex_weights);
  }


  bool isUnit = true;
  for (int a=0; a<orig.GetData().size(); a++) {
    if (abs(orig.GetData()[a] - 1) > 2 * epsilon) {
      isUnit = false;
    }
  }

  float stallThreshold = 0.999;

  int n = A.Rows();

  std::cout << graphPath << std::endl << "|V|: " << n << " |E|: " << A.GetIndices().size()/2 << " avg deg: " << A.GetIndices().size()/2/n << " | is unit: " << isUnit << std::endl;
  /* Partition begin */
  linalgcpp::Timer timer (linalgcpp::Timer::Start::True);
  //SparseMatrix P_T = partition::partition(A, false, true, stallThreshold, 1, false);
  SparseMatrix P_T = partition::partitionBase(A, stallThreshold);
  //SparseMatrix P_T = partition::partition(A, false, true, 1.0, 1, false);
  timer.Click();
  double partitionTime = timer[0];
  // compute modularity
  double partitionModularity = partition::modularity(orig, P_T);

  std::cout << "partition modularity:  \033[1;36m" << partitionModularity << "\033[0m" << std::endl;
  if (printTimes) {
    std::cout << "partition time: \033[1;31m" << partitionTime << "\033[0m" << std::endl;
  }



  timer.Click();
  std::string binPathLouvain = graphPathPrefix + ".bin";
  std::string weightsPathLouvain = graphPathPrefix + ".weights";
  std::string treePathLouvain = graphPathPrefix + ".tree";
  system(("lib/Community_latest/community " + binPathLouvain + " -l -1 -w " + weightsPathLouvain + " > " + graphPathPrefix + ".tree").c_str());
  //system(("lib/Community_latest/hierarchy " + treePathLouvain).c_str());
  timer.Click();
  double louvainTime = timer[2];
  SparseMatrix P_T_louvain = readLouvainAggregates(treePathLouvain);
  double louvainModularity = partition::modularity(orig, P_T_louvain);

  std::cout << "louvain modularity:    \033[1;36m" << louvainModularity << "\033[0m" << std::endl;
  if (printTimes) {
    std::cout << "louvain time: \033[1;31m" << louvainTime << "\033[0m" << std::endl;
  }

  bool doEquiv = false;
  if (doEquiv) {
    long partitionCount=0;
    long partitionTotal=0;
    long louvainCount=0;
    long louvainTotal=0;
    
    std::vector<int> partitionComm = P_T.Transpose().GetIndices();
    std::vector<int> louvainComm = P_T_louvain.Transpose().GetIndices();

    auto PI = P_T.GetIndptr();
    auto PJ = P_T.GetIndices();
    auto LI = P_T_louvain.GetIndptr();
    auto LJ = P_T_louvain.GetIndices();

    for (int A=0; A<P_T.Rows(); A++) {
      long m = PI[A+1] - PI[A];
      partitionTotal += m * (m - 1) / 2;
    }
    
    for (int A=0; A<P_T_louvain.Rows(); A++) {
      long m = LI[A+1] - LI[A];
      louvainTotal += m * (m - 1) / 2;
    }
    
    for (int A=0; A<P_T.Rows(); A++) {
      for (int k=PI[A]; k<PI[A+1]; k++) {
	int i = PJ[k];
	for (int k2=k+1; k2<PI[A+1]; k2++) {
	  int j = PJ[k2];
	  //assert(partitionComm[i] == partitionComm[j]);
	  if (louvainComm[i] == louvainComm[j]) {
	    partitionCount++;
	  }
	}
      }
    }
    //assert(partitionTotal == partitionTotal2);
    
    for (int A=0; A<P_T_louvain.Rows(); A++) {
      for (int k=LI[A]; k<LI[A+1]; k++) {
	int i = LJ[k];
	for (int k2=k+1; k2<LI[A+1]; k2++) {
	  int j = LJ[k2];
	  //louvainTotal++;
	  //assert(louvainComm[i] == louvainComm[j]);
	  if (partitionComm[i] == partitionComm[j]) {
	    louvainCount++;
	  }
	}
      }
    }

    std::cout << (1.0 * partitionCount) / partitionTotal << " " << (1.0 * louvainCount) / louvainTotal << std::endl;
  }

  if (doHeuristics) {
    timer.Click();
    //SparseMatrix P_T_heur = partition::partition(A, false, true, stallThreshold, 2, true);
    SparseMatrix P_T_heur = partition::partitionBase2(A, stallThreshold);
    timer.Click();
    double partitionHeurTime = timer[4];
    // compute modularity
    double partitionHeurModularity = partition::modularity(orig, P_T_heur);
    std::cout << "partition' modularity: \033[1;36m" << partitionHeurModularity << "\033[0m" << std::endl;
    if (printTimes) {
      std::cout << "partition' time: \033[1;31m" << partitionHeurTime << "\033[0m" << std::endl;
    }

    std::cout << "partition: " << P_T.Rows() << " | partition': " << P_T_heur.Rows() << " | louvain: " << P_T_louvain.Rows() << std::endl;

  }

  if (hasTrueCommunities) {
    // for table communities
    // std::vector<int> trueComm = linalgcpp::ReadTable(graphPathPrefix + ".cmty").Transpose().GetIndices();
    // for vector communities
    SparseMatrix P_T_true = readVectorCommunities(graphPathPrefix + ".cmty");
    std::vector<int> partitionComm = P_T.Transpose().GetIndices();
    std::vector<int> louvainComm = P_T_louvain.Transpose().GetIndices();
    std::vector<int> trueComm = P_T_true.Transpose().GetIndices();
    //assert (partitionCommunities.size() == trueCommunities.size());
    double trueModularity = partition::modularity(orig, P_T_true);

    std::cout << "true modularity: " << trueModularity << std::endl;

    std::cout << std::endl;
    
    std::cout << "Sanity check: [part: " << partitionComm.size() << " | louvain: " << louvainComm.size() << " | true: " << trueComm.size() << "]" << std::endl;
    
    int partitionCorrect = 0;
    int louvainCorrect = 0;
    int total = 0;
    int totalPartition = 0;
    int totalLouvain = 0;

    //P_T_true.Print();
    
    for (int i=0; i<n; i++) {
      for (int j=i+1; j<n; j++) {
    	if (trueComm[i] == trueComm[j]) {
    	  if (partitionComm[i] == partitionComm[j]) {
    	    partitionCorrect++;
    	  }
    	  if (louvainComm[i] == louvainComm[j]) {
    	    louvainCorrect++;
    	  }
	  total++;
    	}
	if (partitionComm[i] == partitionComm[j]) {
	  totalPartition++;
	}
	if (louvainComm[i] == louvainComm[j]) {
	  totalLouvain++;
	}
	
      }
    }
    std::cout << "partition similarity: " << (1.0 * partitionCorrect) / (1.0 * total) << "/" << (1.0 * partitionCorrect) / (1.0 * totalPartition) << std::endl;
    std::cout << "louvain similarity: " << (1.0 * louvainCorrect) / (1.0 * total) << "/" << (1.0 * partitionCorrect) / (1.0 * totalLouvain) << std::endl;
  }

  std::cout << std::endl << std::endl << std::endl;

}


int main (int argc, char* argv[]) {

  std::string srcpath = "graphdata/";

  // (shoulduse, pathtograph.txt, format, hascommunities, needs to be symmetrized)
  std::vector<std::tuple<bool, std::string, bool, bool>> entries = {
    //std::make_tuple(true, "ENZYMES_g30",                       false, true), // |V| = 35,   |E| = 60
    //std::make_tuple(true, "karate",                            false, true), // |V| = 35,   |E| = 78
    //std::make_tuple(true, "basketball",                        false, true), // |V| = 100,  |E| = 600
    std::make_tuple(true, "lp_nug06",                          false, false), // |V| = 500,  |E| = 2200
    std::make_tuple(true, "web-google",                        false, false), // |V| = 1.3k, |E| = 1.4k
    std::make_tuple(true, "bio-CE-LC",                         false, false), // |V| = 1400, |E| = 1700
    //std::make_tuple(true, "RO_edges",                          false, true), // |V| = 4k,   |E| = 88k
    //std::make_tuple(true, "HU_edges",                          false, true), // |V| = 4k,   |E| = 88k
    //std::make_tuple(true, "HR_edges",                          false, true), // |V| = 4k,   |E| = 88k
    //std::make_tuple(true, "Email-EuAll",                       false, true), // |V| = 265k, |E| = 365k
    std::make_tuple(true, "roadNet-CA",                        false, false), // |V| = 2M,   |E| = 2.8M
    std::make_tuple(true, "roadNet-PA",                        false, false), // |V| = 1M,   |E| = 1.5M
    std::make_tuple(true, "roadNet-TX",                        false, false), // |V| = 1.4M, |E| = 1.9M
    //std::make_tuple(true, "WikiTalk",                          false, true), // |V| = 2.4M, |E| = 4.7M

    //std::make_tuple(true, "mat_3d.n145_theta0.523599_eps0.01", false, true), // |V| = 145,  |E| = 800
    //std::make_tuple(true, "mat_3d.n867_theta0.523599_eps0.01", false, true), // |V| = 867,  |E| = 5500
    //std::make_tuple(true, "mat_n185_theta0.523599_eps0.01",    false, true), // |V| = 185,  |E| = 600
    std::make_tuple(true, "mat_n697_theta0.523599_eps0.01",    false, false), // |V| = 700,  |E| = 2350
    std::make_tuple(true, "mat_n2705_theta0.523599_eps0.01",   false, false), // |V| = 2700, |E| = 9300

    //std::make_tuple(true, "artist_edges",                      false, true), // |V| = 50k,  |E| = 450k
    std::make_tuple(true, "athletes_edges",                    false, false), // |V| = 14k,  |E| = 56k
    //std::make_tuple(true, "company_edges",                     false, true), // |V| = 14k,  |E| = 38k
    //std::make_tuple(true, "government_edges",                  false, true), // |V| = 7k,   |E| = 51k
    //std::make_tuple(true, "new_sites_edges",                   false, true), // |V| = 28k,  |E| = 129k
    std::make_tuple(true, "politician_edges",                  false, false), // |V| = 6k,   |E| = 26k
    //std::make_tuple(true, "public_figure_edges",               false, true), // |V| = 11k,  |E| = 44k
    std::make_tuple(true, "tvshow_edges",                      false, false), // |V| = 4k,   |E| = 12k
    std::make_tuple(true, "facebook_combined",                 false, false), // |V| = 4k,   |E| = 88k

    
    std::make_tuple(true, "email-Eu-core",                     true, false), // |V| = 1k,   |E| = 25k
    std::make_tuple(true, "com-youtube",                       false, false), // |V| = 1.1M, |E| = 3M

    std::make_tuple(true, "road_germany-osm",                  false, false), // |V| = 11.1M, |E| = 12.3M
    std::make_tuple(true, "delaunay_n24",                      false, false), // |V| = 16.8M  |E| = 50M
    std::make_tuple(true, "com-lj",                            false, false), // |V| = 4M,     |E| = 34.7M
    std::make_tuple(false, "", false, false)
  };

  for (int i=0; i<entries.size(); i++) {
    bool use = std::get<0>(entries[i]);
    std::string graphPath = std::get<1>(entries[i]);
    bool hasTrueCommunities = std::get<2>(entries[i]) && doCommunities;
    bool symmetrize = std::get<3>(entries[i]);

    if (symmetrize) {
      std::cout << graphPath << std::endl;
      SparseMatrix orig = linalgcpp::ReadCooList(srcpath + graphPath + ".edges", true);
      linalgcpp::WriteCooList(orig, srcpath + graphPath + "2.edges", false);
      system(("lib/Community_latest/convert -i " + srcpath + graphPath + "2.edges -o " + srcpath + graphPath + ".bin -w " + srcpath + graphPath + ".weights").c_str());      
    }

    if (use) {
      runTest (srcpath + graphPath, "coolist", hasTrueCommunities);
    }
  }
  
}






