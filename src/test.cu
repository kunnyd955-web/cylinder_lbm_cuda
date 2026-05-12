#include <stdio.h>

__global__ void hello_gpu() {
    printf("Hello from thread %d\n", threadIdx.x);
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("显存: %.1f GB\n", prop.totalGlobalMem / 1e9);
    
    hello_gpu<<<1, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
