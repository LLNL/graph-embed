
// Copyright 2019 Lawrence Livermore National Security. 
// Produced at the Lawrence Livermore National Laboratory.
// LLNL-CODE-781679. All Rights reserved. See file LICENSE for details.
//
// This file is part of graph-embed. For more information and source code
// availability, see github.com/LLNL/graph-embed
//
// SPDX-License-Identifier: LGPL-2.1


#include "matrixutils.hpp"

namespace partition {

  SparseMatrix identity(int n){
    /*std::vector<int> nI (n+1);
      nI[0] = 0;
      std::vector<int> nJ (n);
      std::vector<double> nD (n, 1.0);

      for (int i=0; i<n; i++) {
      nI[i+1] = i+1;
      nJ[i] = i;
      }
    
      return SparseMatrix(nI, nJ, nD, n, n);*/
    return SparseMatrix(std::vector<double>(n, 1.0));
  }

  SparseMatrix toLaplacian(const SparseMatrix& A) {
    int numRows = A.Rows();
    const std::vector<int>& I = A.GetIndptr();
    const std::vector<int>& J = A.GetIndices();
    const std::vector<double>& D = A.GetData();

    std::vector<int> nI (numRows+1);
    for (int i=1; i<numRows+1; i++) 
      {nI[i] = I[i]+i;}
    std::vector<int> nJ(A.nnz()+numRows);
    std::vector<double> nD(A.nnz()+numRows);

    int count=0;
    for (int i=0; i<numRows; i++) {
      bool passed=false;
      for (int k=I[i]; k<I[i+1]; k++) {
	if (!passed && J[k]>i){
	  double sum=0; 
	  for (int k0=I[i]; k0<I[i+1]; k0++) 
	    {sum+=D[k0];}
	  nJ[k+count] = i; 
	  nD[k+count] = sum;
	  count++;
	  passed=true;
	}
	nJ[k+count] = J[k]; 
	nD[k+count] = -D[k];
      }
      if (!passed) {
	double sum=0; 
	for (int k0=I[i]; k0<I[i+1]; k0++) 
	  {sum+=D[k0];}

	nJ[I[i+1]+count] = i; 
	nD[I[i+1]+count] = sum;
	count++;
      }
    }
    return SparseMatrix(nI, nJ, nD, numRows, A.Cols());
  }

  SparseMatrix fromLaplacian(const SparseMatrix& L) {
    int numRows = L.Rows();
    const std::vector<int>& I = L.GetIndptr();
    const std::vector<int>& J = L.GetIndices();
    const std::vector<double>& D = L.GetData();

    std::vector<int> nI (numRows+1);
    std::vector<int> nJ (L.nnz() - numRows);
    std::vector<double> nD (L.nnz() - numRows);
    for (int i=1; i<numRows+1; i++) 
      {nI[i] = I[i]-i;}

    int count=0;
    for (int i=0;i<numRows; i++) {
      bool passed=false;
      for(int k=nI[i]; k<nI[i+1]; k++) {
	if (!passed && J[k+count]>=i) {
	  count++;
	  passed=true;
	}
	nJ[k] = J[k+count]; nD[k] = -D[k+count];
      }
      if (!passed)
	{count++;}
    }
    return SparseMatrix(nI, nJ, nD, numRows, L.Cols());
  }

}
