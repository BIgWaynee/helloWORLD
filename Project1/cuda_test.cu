// cuda_test.cu
#include "cuda_test.cuh"


void GpuDeviceInfo()
{
	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
	if (error_id != cudaSuccess)
	{
		std::cout << "no GUP can query,configure may have error!\n";
		return;
	}

	for (int dev = 0; dev < deviceCount; dev++)
	{
		cudaSetDevice(dev);
		cudaDeviceProp devProp;
		error_id = cudaGetDeviceProperties(&devProp, dev);
		if (error_id != cudaSuccess)
		{
			std::cout << "GetDeviceProperties error !" << std::endl;
		}
		else
		{
			std::cout << "using GPU device " << dev << ":" << devProp.name << std::endl;
			std::cout << "number of SM: " << devProp.multiProcessorCount << std::endl;
			std::cout << "shared momery size of one thread block: " << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
			std::cout << "max thread number of one thread block: " << devProp.maxThreadsPerBlock << std::endl;
			std::cout << "max thread number of one EM: " << devProp.maxThreadsPerMultiProcessor << std::endl;
			std::cout << "max thread wraps of one EM: " << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
		}
	}
}

// ���������ӷ�kernel��grid��block��Ϊһά
__global__ void gpu_add(float* x, float * y, float* z, int n)
{
	// ��ȡȫ������
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// ����
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
	{
		z[i] = x[i] + y[i];
	}
}


void test_gpuAdd()
{
	int n = 100;
	int nBytes = n * sizeof(float);
	float *a = (float *)malloc(nBytes);
	float *b = (float *)malloc(nBytes);
	float *c = (float *)malloc(nBytes);
	for (int i = 0; i < n; i++)
	{
		a[i] = 10.0f;
		b[i] = 20.0f;
		c[i] = 0.f;
	}

	float *dx, *dy, *dz;
	cudaMalloc(&dx, nBytes);
	cudaMalloc(&dy, nBytes);
	cudaMalloc(&dz, nBytes);

	cudaMemcpy(dx, a, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dy, b, nBytes, cudaMemcpyHostToDevice);
	//cudaMemcpy(dz, c, nBytes, cudaMemcpyHostToDevice);

	dim3 blockSize(256);
	dim3 gridSize = (n + blockSize.x - 1) / blockSize.x;

	gpu_add << <gridSize, blockSize >> > (dx, dy, dz, n);
	cudaMemcpy(c, dz, nBytes, cudaMemcpyDeviceToHost);

	float maxDiff = .0f;
	for (int i = 0; i < n; i++)
	{
		if (std::abs(c[i] - 30.0) > maxDiff)
			maxDiff = c[i] - 30.0f;
	}
	std::cout << "max difference: " << maxDiff << std::endl;


	cudaFree(dx);
	cudaFree(dy);
	cudaFree(dz);
	free(a);
	free(b);
	free(c);
}


//�ں˺���
__global__ void rgb2grayincuda(uchar3 * const d_in, unsigned char * const d_out,
	uint32_t imgheight, uint32_t imgwidth)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx < imgwidth && idy < imgheight)  //�е��̻߳��ܵ�ͼ������ȥ����ִ�м���
	{
		uchar3 rgb = d_in[idy * imgwidth + idx];
		d_out[idy * imgwidth + idx] = 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
	}
}

//���ڶԱȵ�CPU���д���
void rgb2grayincpu(unsigned char * const d_in, unsigned char * const d_out,
	uint32_t imgheight, uint32_t imgwidth)
{
	for (int i = 0; i < imgheight; i++)
	{
		for (int j = 0; j < imgwidth; j++)
		{
			d_out[i * imgwidth + j] = 0.299f * d_in[(i * imgwidth + j) * 3]
				+ 0.587f * d_in[(i * imgwidth + j) * 3 + 1]
				+ 0.114f * d_in[(i * imgwidth + j) * 3 + 2];
		}
	}
}

void test_bgr2gray(int flag)
{
	using namespace std;

	const uint32_t imgheight = 960;
	const uint32_t imgwidth = 480;
	uchar3 *srcData = (uchar3 *)malloc(imgheight*imgwidth* sizeof(uchar3));
	for (int row = 0; row < imgheight; row++)
	{
		uchar3 *rowData = srcData + row*imgwidth;
		for (int col = 0; col < imgwidth; col++)
		{

			uchar3 *pixel = rowData + col;
			pixel->x = col % 255;
			pixel->y = row % 255;
			pixel->z = (row + col) % 255;
		}
	}

	uchar3 *d_in;   //�������ͣ�3��uchar
	unsigned char *d_out;

	//���ȷ���GPU�ϵ��ڴ�
	cudaMalloc((void**)&d_in, imgheight*imgwidth * sizeof(uchar3));
	cudaMalloc((void**)&d_out, imgheight*imgwidth * sizeof(unsigned char));

	//�����������ݿ�����GPU��
	cudaMemcpy(d_in, srcData, imgheight*imgwidth * sizeof(uchar3), cudaMemcpyHostToDevice);

	//ÿ���̴߳���һ������
	dim3 threadsPerBlock(32, 32);
	dim3 blocksPerGrid((imgwidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(imgheight + threadsPerBlock.y - 1) / threadsPerBlock.y);

	clock_t start, end;
	start = clock();

	//�����ں�
	rgb2grayincuda << <blocksPerGrid, threadsPerBlock >> >(d_in, d_out, imgheight, imgwidth);
	//static int flag = 0;
	cout << flag << endl;

	//ִ���ں���һ���첽�����������Ҫͬ���Բ���׼ȷʱ��
	//cudaDeviceSynchronize();
	end = clock();

	//printf("cuda exec time is %.8f\n", (double)(end - start) / double(CLOCKS_PER_SEC)*1000);

	//������������
	cudaMemcpy(srcData, d_out, imgheight*imgwidth * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	free(srcData);
	//�ͷ��Դ�
	cudaFree(d_in);
	cudaFree(d_out);

	//imshow("grayImage", grayImage);

}

