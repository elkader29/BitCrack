#include "cudabridge.h"


__global__ void keyFinderKernel(int points, int compression);
__global__ void keyFinderKernelWithDouble(int points, int compression);

void callKeyFinderKernel(int blocks, int threads, int points, bool useDouble, int compression)
{
	if(useDouble) {
		keyFinderKernelWithDouble <<<blocks, threads >>>(points, compression);
	} else {
		keyFinderKernel <<<blocks, threads>>> (points, compression);
	}
	waitForKernel();
}

__global__ void exportKernel(int mode, unsigned int *startKey, unsigned int *endKey, unsigned int *seed, const unsigned int *basePointsX, const unsigned int *basePointsY);

void callExportKernel(int blocks, int threads, int mode, unsigned int *startKey, unsigned int *endKey, unsigned int *seed, const unsigned int *basePointsX, const unsigned int *basePointsY)
{
    exportKernel<<<blocks, threads>>>(mode, startKey, endKey, seed, basePointsX, basePointsY);
    waitForKernel();
}


void waitForKernel()
{
    // Check for kernel launch error
    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess) {
        throw cuda::CudaException(err);
    }
 
    // Wait for kernel to complete
    err = cudaDeviceSynchronize();
	fflush(stdout);
	if(err != cudaSuccess) {
		throw cuda::CudaException(err);
	}
}