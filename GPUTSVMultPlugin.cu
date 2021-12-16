#include <emmintrin.h>
#include <sys/time.h> 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <iostream>
#include <iomanip>
#include <fstream>

#include "GPUTSVMultPlugin.h"

void GPUTSVMultPlugin::input(std::string file) {
 inputfile = file;
 std::ifstream ifile(inputfile.c_str(), std::ios::in);
 while (!ifile.eof()) {
   std::string key, value;
   ifile >> key;
   ifile >> value;
   parameters[key] = value;
 }
 M = atoi(parameters["M"].c_str());
 N = atoi(parameters["N"].c_str());
 P = atoi(parameters["P"].c_str());
 A = (double*) malloc(N*N*sizeof(double));
 B = (double*) malloc(N*N*sizeof(double));
 C = (double*) malloc(N*N*sizeof(double));
 std::ifstream myinput((std::string(PluginManager::prefix())+parameters["matrix1"]).c_str(), std::ios::in);
 int i;
 for (i = 0; i < M*N; ++i) {
	int k;
	myinput >> k;
        A[i] = k;
 }
 std::ifstream myinput2((std::string(PluginManager::prefix())+parameters["matrix2"]).c_str(), std::ios::in);
 for (i = 0; i < N*P; ++i) {
	int k;
	myinput2 >> k;
        B[i] = k;
 }
}




void GPUTSVMultPlugin::run() {
	double *pA;
	double *pB;
	double *pC;
cudaMalloc((void**)&pA, (M*N)*sizeof(double));
cudaMalloc((void**)&pB, (N*P)*sizeof(double));
cudaMalloc((void**)&pC, (M*P)*sizeof(double));
cudaMemcpy(pA, A, (M*N)*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(pB, B, (N*P)*sizeof(double), cudaMemcpyHostToDevice);
printf("***Mult on %d x %d Matrix on GPU***\n",N,N);
MatMult<<<M,P>>>(pA, pB, pC, M, N, P);
cudaMemcpy(C, pC, (M*P)*sizeof(double), cudaMemcpyDeviceToHost);

cudaFree(pA);
cudaFree(pB);
cudaFree(pC);

}

void GPUTSVMultPlugin::output(std::string file) {
	std::ofstream outfile(file.c_str(), std::ios::out);
        int i, j;
        for (i = 0; i < M; ++i){
            for (j = 0; j < P; ++j){
		outfile << C[i*P+j];//std::setprecision(0) << a[i*N+j];
		if (j != P-1)
			outfile << "\t";
		else
			outfile << "\n";
            }
	}
	free(A);
	free(B);
	free(C);
}



PluginProxy<GPUTSVMultPlugin> GPUTSVMultPluginProxy = PluginProxy<GPUTSVMultPlugin>("GPUTSVMult", PluginManager::getInstance());


