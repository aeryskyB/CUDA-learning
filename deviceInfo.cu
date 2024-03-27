#include <stdio.h>
#include "device_launch_parameters.h"

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of CUDA enabled devices: %d\n", deviceCount);
    cudaDeviceProp dProps;
    cudaError_t e;
    for (int i = 0; i < deviceCount; i++) {
        e = cudaGetDeviceProperties(&dProps, i);
        if (e == cudaSuccess) {
            printf("For device %d:\n", i);
            printf("Max. threads per block: %d\n", dProps.maxThreadsPerBlock);
            printf("#SM: %d\n", dProps.multiProcessorCount);
            printf("Maximum #blocks per SM: %d\n", dProps.maxBlocksPerMultiProcessor);
            printf("Threads per SM: %d\n", dProps.maxThreadsPerMultiProcessor);
            printf("Warp size in threads: %d\n", dProps.warpSize);
            printf("#Registers available per block/SM: %d\n", dProps.regsPerBlock);
            printf("Clock rate: %d\n", dProps.clockRate);
        } else {
            printf("Error getting properties for device %d!\n", i);
        }
        printf("\n");
    }
}