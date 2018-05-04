#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

__global__ void squareMultiply(int *A, int *B, int *result, int *nDim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    int dim = *nDim;

    if (row < dim && col < dim) {
        int sum = 0;
        #pragma unroll
        for (int i = 0; i < dim; i++) {
            sum += A[row * dim + i] * B[i * dim + i];
        }
        result[row * dim + col] = sum;
    }
//     __syncthreads();   
}

void runKernel() {
    srand(0);
    int maxLength = 10;
   
    int nLength = (rand() % maxLength) + maxLength; // make sure the length is at least the size of maxLength
    int sizeOfMatrix = nLength * nLength;

    printf("%d\n",sizeOfMatrix);
    // create arrays for algorithm
    int *A = (int*) malloc (sizeof(int) * sizeOfMatrix);
    int *B = (int*) malloc(sizeof(int) * sizeOfMatrix);
    int *results = (int*) malloc(sizeof(int) * sizeOfMatrix);

    printf("about to fill\n");
    // fill arrays with random numbers
    for (int i = 0; i < sizeOfMatrix; i++) {
        A[i] = rand() % 1000;
        B[i] = rand() % 1000;
    }

    printf("finished filling\n");
    int *d_A;
    if (cudaMalloc((void**) &d_A, sizeof(A)) != cudaSuccess) {
        printf("Error allocating for matrix A on gpu");
        exit(-1);
    }

    int *d_B;
    
    if (cudaMalloc((void**) &d_B, sizeOfMatrix * sizeof(int)) != cudaSuccess) {
        printf("Error allocating for matrix B on gpu");
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
    dim3 threadsPerBlock(512, 512); // can have up to 1024 threads per block on our gpu

    int blockDimSize = ceil(sizeOfMatrix / threadDim); 
    dim3 blocks(blockDimSize, blockDimSize);

//    squareMultiply<<<blockDimSize,threadsPerBlock>>>(d_A, d_B, d_results, d_nDim);
//    cudaDeviceSynchronize();

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
