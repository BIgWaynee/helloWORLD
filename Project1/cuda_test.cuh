//cuda_test.cuh
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdint.h>
#include <vector>
#include <time.h>

void GpuDeviceInfo();
void test_gpuAdd();
__global__ void rgb2grayincuda(uchar3 * const d_in, unsigned char * const d_out,
	uint32_t imgheight, uint32_t imgwidth);
void rgb2grayincpu(unsigned char * const d_in, unsigned char * const d_out,
	uint32_t imgheight, uint32_t imgwidth);
void test_bgr2gray(int flag = 0);
