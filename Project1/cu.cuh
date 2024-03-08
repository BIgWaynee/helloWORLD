#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


namespace basicDem {
	class cGpuCal {
	public: 


		double* c;

		double* d_a;
		double* d_b;
		double* d_c;

		void sumMatrix();
		friend void __global__ global_sumMatrix(double*a, double* b, double*c, int n);
	};
}

inline void getResultFormcGpuCal() {
	basicDem::cGpuCal ob{};
	ob.sumMatrix();
	double* temp = ob.c;
	for (int i{ 0 }; i < 3; i++) {
	std::cout << temp[i] << std::endl;
	}
}