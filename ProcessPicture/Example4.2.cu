
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

cudaError_t addWithCuda(float *Picture, int m, int n);

__global__ void PictureKernel(float *d_Pin, float *d_Pout, int m, int n)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	if((Row < m) && (Col < n)) {
		d_Pout[Row * n + Col] = 2 * d_Pin[Row * n + Col];
	}
}

int main()
{
	// create a picture
	int i, j;
	float picture[9][12];
	for(i = 0; i < 9; i++) {
		for(j = 0; j < 12; j++) {
			picture[i][j] = i * 12 + j;
		}
	}
	// mapping picture data into 1D array
	float *picture_1D;
	picture_1D = (float *)malloc(9*12*sizeof(float));	
	for(i = 0; i < 9; i++) {
		for(j = 0; j < 12; j++) {
			picture_1D[i*12 + j] = picture[i][j];
		}
	}

	// before calling picturekernel
	for(i = 0; i < 9; i++) {
		for(j = 0; j < 12; j++) {
			printf("%1.0f  ", picture_1D[i*12 + j]);
		}
		printf("\n");
	}    

    // call the addwithcuda function.
    cudaError_t cudaStatus = addWithCuda(picture_1D, 9, 12);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	printf("\n*******============================================================*********\n");

	// after calling picturekernel
	for(i = 0; i < 9; i++) {
		for(j = 0; j < 12; j++) {
			printf("%1.0f  ", picture_1D[i*12 + j]);
		}
		printf("\n");
	} 

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(float *Picture, int m, int n)
{
    float *d_Pin, *d_Pout;
	int Psize = m * n * sizeof(float);
	cudaMalloc((void**)&d_Pin, Psize);
	cudaMemcpy(d_Pin, Picture, Psize, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_Pout, Psize);
    cudaError_t cudaStatus;

	dim3 threadsPerBlocks(16, 16, 1);
	dim3 blocksPerGird(ceil(n/16.0), ceil(m/16.0), 1);
	

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    PictureKernel<<<blocksPerGird, threadsPerBlocks>>>(d_Pin, d_Pout, m, n);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(Picture, d_Pout, Psize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(d_Pin);
    cudaFree(d_Pout);
        
    return cudaStatus;
}
