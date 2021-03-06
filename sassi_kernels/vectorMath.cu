#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

__global__ void vectorAdd(int *A, int *B, int *result, int *length) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId < *length) {
        result[threadId] = A[threadId] + B[threadId];
    }
    return;
}

void runVectorAdd() {
    srand(0);
    int maxLength = 10000;
    int length = (rand() % maxLength) + maxLength; // make sure the length is at least the size of maxLength
    
    // create arrays for algorithm
    int *A = (int*) malloc (sizeof(int) * length);
    int *B = (int*) malloc(sizeof(int) * length);
    int *results = (int*) malloc(sizeof(int) * length);

    // fill arrays with random numbers
    for (int i = 0; i < length; i++) {
        A[i] = rand() % 1000;
        B[i] = rand() % 1000;
    }
  
    int *d_A;

    if (cudaMalloc((void**) &d_A, length * sizeof(int)) != cudaSuccess) {
        printf("Error allocating for vector A on gpu");
        exit(-1);
    }

    if (cudaMemcpy(d_A, A, length * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("Error copying vector A onto gpu");
        exit(-1);
    }
    
    int *d_B;
    if (cudaMalloc((void**) &d_B, length * sizeof(int)) != cudaSuccess) {
        printf("Error allocating for vector B on gpu");
        exit(-1);
    }

    if (cudaMemcpy(d_B, B, length * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("Error copying vector B onto gpu");
        exit(-1);
    }

    int *d_results;

    if (cudaMalloc((void**) &d_results, length * sizeof(int)) != cudaSuccess) {
        printf("Error allocating for results vector on gpu");
        exit(-1);
    }

    if (cudaMemcpy(d_results, results, length * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("Error copying results vector onto gpu");
        exit(-1);
    }

    int *d_length;

    if (cudaMalloc((void**) &d_length, sizeof(int)) != cudaSuccess) {
        printf("Error allocating for length variable on gpu");
        exit(-1);
    }

    if (cudaMemcpy(d_length, &length, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("Error copying length variable onto gpu");
        exit(-1);
    }
    
    int numThreadsPerBlock = min(1024, length); // can have up to 1024 threads per block on our gpu
    int numBlocks = (length + numThreadsPerBlock - 1) / numThreadsPerBlock;

    vectorAdd<<<numBlocks,numThreadsPerBlock>>>(d_A, d_B, d_results, d_length);
    cudaDeviceSynchronize();
    
    std::cout << "The last error was: ";
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;    
    
    // get results back
    if (cudaMemcpy(results, d_results, length * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("error getting results back from device");
        exit(-1);
    }

    // print results
    for (int i = 0; i < length; i++) {
        printf("%d", results[i]);
        if (i+1 != length) {
            printf(", ");
        }
    }
}

int main (int argc, char** argv) {
    runVectorAdd();
    return 0;
}

