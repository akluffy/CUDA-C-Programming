// cuda Matrix Addition using A[i][j] and (*a)[j] declaration and 1 dimension blocks x . 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

const int M = 10, N = 8;

cudaError_t addWithCuda(float A[M][N], float B[M][N], float C[M][N]);

__global__ void addKernel(float A[M][N], float B[M][N], float C[M][N])
{
    int i = threadIdx.x / N;	 // columns
	int j = threadIdx.x % N;     // rows
    C[i][j] = A[i][j] + B[i][j];
}

int main()
{
	// allocation in host
    float a[M][N];
	float b[M][N];
	float c[M][N];

	// initialization original data in host
	for(int i = 0; i < M; i++) {
		for(int j = 0; j < N; j++) {
			a[i][j] = i;
			b[i][j] = j;	
		}
	}

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(a, b, c);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // display the result
	for(int i = 0; i < M; i++) {
		for(int j = 0; j < N; j++) {
			printf(" %1.f ", a[i][j]);
		}
		printf("\n");
	}
	
	printf("\t Plus \t\n");

	for(int i = 0; i < M; i++) {
		for(int j = 0; j < N; j++) {
			printf(" %1.f ", b[i][j]);
		}
		printf("\n");
	}

	printf("========================\n");

	for(int i = 0; i < M; i++) {
		for(int j = 0; j < N; j++) {
			printf(" %1.f ", c[i][j]);
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
cudaError_t addWithCuda(float A[M][N], float B[M][N], float C[M][N])
{
    // create device varibles
	float (*d_a)[N];
    float (*d_b)[N];
    float (*d_c)[N];
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&d_a, M * N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&d_b, M * N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&d_c, M * N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(d_a, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(d_b, B, M * N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
	dim3 BlocksPerGrid(1, 1, 1);
	dim3 ThreadsPerBlock(M * N, 1, 1);
    addKernel<<<BlocksPerGrid, ThreadsPerBlock>>>(d_a, d_b, d_c); 

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
    cudaStatus = cudaMemcpy(C, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return cudaStatus;
}
