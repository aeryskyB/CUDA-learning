#include <stdio.h>
#include "device_launch_parameters.h"

// take at least 16 and in power of 2
#define WIDTH 64

__global__
void matMulKernel(int *A, int *B, int *C, int width) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (row < width && col < width) {
        int val = 0;
        for (int i = 0; i < width; i++)
            val += A[row*width + i] * B[i*width + col];
        C[row*width + col] = val;
    }
}

void matMul(int *A_h, int *B_h, int *C_h, int width) {
    int size = width * width * sizeof(int);
    int *A_d, *B_d, *C_d;
    int grid_width = WIDTH / 16;
    int block_width = WIDTH / grid_width;

    // allocate device memory
    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size);
    cudaMalloc((void **)&C_d, size);

    // copy data from host to device
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(grid_width, grid_width, 1);
    dim3 dimBlock(block_width, block_width, 1);
    // call kernel
    matMulKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, width);

    // free device memory
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    // consider 3 2D matrices
    int A[WIDTH][WIDTH], B[WIDTH][WIDTH], C[WIDTH][WIDTH];
    
    // ?: Our target is C = A*B  [* -> matrix multiplication]

    // fill A, and B as you want
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++) {
            A[i][j] = i*j;
            B[i][j] = i+j;
        }
    }

    printf("A:\n");
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++)
            printf("%d ", A[i][j]);
        printf("\n");
    }

    printf("B:\n");
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++)
            printf("%d ", B[i][j]);
        printf("\n");
    }

    // define 1D counterparts
    int A_f[WIDTH*WIDTH], B_f[WIDTH*WIDTH], C_f[WIDTH*WIDTH];
    // flatten A, and B
    for (int i = 0; i < WIDTH*WIDTH; i++) {
        A_f[i] = A[i/WIDTH][i%WIDTH];
        B_f[i] = B[i/WIDTH][i%WIDTH];
    }

    // perform multiplication
    matMul(A_f, B_f, C_f, WIDTH);

    for (int i = 0; i < WIDTH*WIDTH; i++) {
        C[i/WIDTH][i%WIDTH] = C_f[i]; 
    }

    printf("C:\n");
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++)
            printf("%d ", C[i][j]);
        printf("\n");
    }

    return 0;
}