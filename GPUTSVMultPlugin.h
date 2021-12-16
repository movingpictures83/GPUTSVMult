#ifndef GPUTSVMULTPLUGIN_H
#define GPUTSVMULTPLUGIN_H

#include "Plugin.h"
#include "PluginProxy.h"
#include <string>
#include <map>

class GPUTSVMultPlugin : public Plugin {

	public:
		void input(std::string file);
		void run();
		void output(std::string file);
	private:
                std::string inputfile;
		std::string outputfile;
		double* A;
		double* B;
		double* C;
		int M;
		int N;
		int P;
                std::map<std::string, std::string> parameters;
};
__global__ void MatMult(double* A, double* B, double* C, int M, int N, int P){
               int i = blockIdx.x;
               int j = threadIdx.x;
               int k;
               double Pvalue = 0;

               for(k = 0; k < N; k++){
                   Pvalue += A[i*N+k] * B[k*P+j];
               }

               C[i*P+j] = Pvalue;

}

#endif
