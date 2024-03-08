#include "cu.cuh"
#include"cuda_runtime.h"
void __global__  basicDem::global_sumMatrix(double*a , double*b , double*c , int n) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) return;
	c[idx] = a[idx] + b[idx];
	printf("c[%d]: %f\n", idx, c[idx]);
}


//unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
//
//if (idx >= n) return;
//
//c[idx] = a[idx] + b[idx];

void basicDem::cGpuCal::sumMatrix() {
	double temp_a[3] = { 1.1 , 2.2 , 3.3 };
	double temp_b[3] = { 1.0 ,1.0 ,1.0 };
	
	cudaMalloc((void**)&d_a, sizeof(double) * 3);
	cudaMalloc((void**)&d_b, sizeof(double) * 3);
	cudaMalloc((void**)&d_c, sizeof(double) * 3);

	cudaMemcpy(d_a, temp_a, sizeof(double) * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, temp_b, sizeof(double) * 3, cudaMemcpyHostToDevice);

	int dimx = 3; 
	int dimy = 1;
	dim3 grid_0(dimx, dimy);
	dim3 block_0 = 1;
	global_sumMatrix << <grid_0, block_0 >> > (d_a, d_b, d_c, 3);
	cudaHostAlloc((void**)&c, sizeof(double) * 3, cudaHostAllocDefault);

	cudaMemcpy(c, d_c, sizeof(double) * 3, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	std::cout << "In sumMatrix " << std::endl;
	std::cout << "c[0] : " << c[0] << std::endl;
	std::cout << "c[1] : " << c[1] << std::endl;
	std::cout << "c[2] : " << c[2] << std::endl;
}





