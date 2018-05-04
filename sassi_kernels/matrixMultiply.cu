#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

__global__ void squareMultiply(int *A, int *B, int *result, int *nDim) {
    // get the indices for each matrix to start at
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int dim = *nDim;
    
    if (row < dim && col < dim) {
        // calculate a single result 
        int sum = 0;
        for (int i = 0; i < dim; i++) {
            sum += A[row * dim + i] * B[i * dim + col];
	}
        result[row * dim + col] = sum;
    }
}

void runKernel() {
    srand(0);
    int maxLength = 20;  // since we're using ints, keep this small to avoid overflow
   
    int nLength = (rand() % maxLength) + maxLength; // make sure the length is at least the size of maxLength
    int sizeOfMatrix = nLength * nLength;

    // create arrays for algorithm
    int *A = (int*) malloc (sizeof(int) * sizeOfMatrix);
    int *B = (int*) malloc(sizeof(int) * sizeOfMatrix);
    int *results = (int*) malloc(sizeof(int) * sizeOfMatrix);

    // fill arrays with random numbers
    for (int i = 0; i < sizeOfMatrix; i++) {
        A[i] = rand() % 20;
        B[i] = rand() % 20;
    }

    int *d_A;
    if (cudaMalloc((void**) &d_A, sizeOfMatrix * sizeof(int)) != cudaSuccess) {
        printf("Error allocating for matrix A on gpu");
        exit(-1);
    }
   
    if (cudaMemcpy(d_A, A, sizeOfMatrix * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("Error copying matrix A onto gpu");
        exit(-1);
    }

    int *d_B;
    
    if (cudaMalloc((void**) &d_B, sizeOfMatrix * sizeof(int)) != cudaSuccess) {
        printf("Error allocating for matrix B on gpu");
        exit(-1);
    }
    
    if (cudaMemcpy(d_B, B, sizeOfMatrix * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("Error copying matrix B onto gpu");
        exit(-1);
    }

    int *d_results;

    if (cudaMalloc((void**) &d_results, sizeOfMatrix * sizeof(int)) != cudaSuccess) {
        printf("Error allocating for matrix results on gpu");
        exit(-1);
    }
    

    int *d_nDim;
    if (cudaMalloc((void**) &d_nDim, sizeof(int)) != cudaSuccess) {
        printf("Error allocating for length variable on gpu");
        exit(-1);
    }
    
    if (cudaMemcpy(d_nDim, &nLength, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("Error copying length of matrix onto gpu");
        exit(-1);
    }
 
    // set up kernel launch
    int threadDim = min(512, nLength);
    dim3 threadsPerBlock(threadDim,threadDim); // can have up to 1024 threads per block on our gpu

    int blockDimSize = ceil(sizeOfMatrix / threadDim); 
    dim3 blocks(blockDimSize, blockDimSize);

    squareMultiply<<<blockDimSize,threadsPerBlock>>>(d_A, d_B, d_results, d_nDim);
    cudaDeviceSynchronize();

    std::cout << "The last error was: ";
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;     

    //retrieve results
    if (cudaMemcpy(results, d_results, sizeOfMatrix * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("error getting results matrix back from device");
        exit(-1);
    }

    // print results
    for (int i = 0; i < sizeOfMatrix; i++) {
        if (i % nLength == 0) {
            printf("\n");
        }
        printf("%d", results[i]);
        if (i+1 != sizeOfMatrix) {
            printf(", ");
        }
    }
}

int main(int argc, char** argv) {
    runKernel();
    return 0;
}
