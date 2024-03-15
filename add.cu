#include <stdio.h>
#include "device_launch_parameters.h"

__global__
void vecAddKernel(float *A, float *B, float* C, int n) {
    unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float *A_h, float *B_h, float *C_h, int n) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    // ^ Allocate device global memory
    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size);
    cudaMalloc((void **)&C_d, size);

    // ^ Copy data from host -> device
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // ^ Call cuda kernel (function) for sum
    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

    // ^ Copy result data from device to host
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    // ^ Free allocated memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    const int n = 1024;
    float A_h[n], B_h[n], C_h[n];
    for (int i = 0; i < n; i++) {
        A_h[i] = i;
        B_h[i] = i;
    }

    vecAdd(A_h, B_h, C_h, n);

    for (int i = 0; i < n; i++) {
        printf("%f ", C_h[i]);
    }

    return 0;
}