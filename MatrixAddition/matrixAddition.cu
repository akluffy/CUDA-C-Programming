 #include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */

#define M 5
#define N 4

__global__ void MatAdd(float A[M][N], float B[M][N],
                       float C[M][N])
{
    int j = threadIdx.x;
    int i = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

/**
 * Host main routine
 */
int main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

float A[M][N];
float B[M][N];
float C[M][N];

float (*d_A)[N]; //pointers to arrays of dimension N
float (*d_B)[N];
float (*d_C)[N];

for(int i = 0; i < M; i++) {
    for(int j = 0; j < N; j++) {
        A[i][j] = i;
        B[i][j] = j;
    }
}       

//allocation
cudaMalloc((void**)&d_A, (M*N)*sizeof(float));
cudaMalloc((void**)&d_B, (M*N)*sizeof(float));
cudaMalloc((void**)&d_C, (M*N)*sizeof(float));

//copying from host to device
cudaMemcpy(d_A, A, (M*N)*sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_B, B, (M*N)*sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_C, C, (M*N)*sizeof(float), cudaMemcpyHostToDevice);

// Kernel invocation
dim3 threadsPerBlock(N, M);
dim3 numBlocks(N / threadsPerBlock.x, M / threadsPerBlock.y);
MatAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

//copying from device to host
cudaMemcpy(A, (d_A), (M*N)*sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(B, (d_B), (M*N)*sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(C, (d_C), (M*N)*sizeof(float), cudaMemcpyDeviceToHost);

for(int i = 0; i < M; i++) {
	for(int j = 0; j < N; j++) {
		printf(" %1.f ", A[i][j]);
	}
	printf("\n");
}

printf(" pLUS \n");

for(int i = 0; i < M; i++) {
	for(int j = 0; j < N; j++) {
		printf(" %1.f ", B[i][j]);
	}
	printf("\n");
}

printf("====================\n");

for(int i = 0; i < M; i++) {
	for(int j = 0; j < N; j++) {
		printf(" %1.f ", C[i][j]);
	}
	printf("\n");
}


    // Reset the device and exit
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}

