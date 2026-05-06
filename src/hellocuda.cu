#include <stdio.h>

__global__ void hello_kernel() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main() {
    printf("Hello from CPU!\n");
    printf("Launching GPU kernel...\n");

    hello_kernel<<<1, 4>>>();
    cudaDeviceSynchronize();

    printf("Done!\n");
    return 0;
}