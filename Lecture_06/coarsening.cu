#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// Size of the array (32 MB floats)
#define N (1 << 25)
#define BLOCK_SIZE 256
// Thread coarsening factor: Each thread processes 2 or 4 elements
#define COARSE_FACTOR 2

// Warmup Kernel
__global__ void warmup_kernel(float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        b[i] = a[i];
    }
}

// Fine-Grained Kernel: 1 element per thread
__global__ void fine_grained_kernel(float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Simple scale
        b[i] = a[i] * 2.0f;
    }
}

// Coarsened Kernel: COARSE_FACTOR elements per thread
// Grid size is reduced by COARSE_FACTOR
__global__ void coarsened_kernel(float* a, float* b, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * COARSE_FACTOR;
    
    // Process COARSE_FACTOR elements sequentially
    #pragma unroll
    for (int j = 0; j < COARSE_FACTOR; j++) {
        if (i + j < n) {
             b[i + j] = a[i + j] * 2.0f;
        }
    }
}

int main() {
    float *d_a, *d_b;
    size_t size = N * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMemset(d_a, 0, size); // Initialize

    int threads = BLOCK_SIZE;
    
    // Warmup
    int blocks_warmup = (N + threads - 1) / threads;
    warmup_kernel<<<blocks_warmup, threads>>>(d_a, d_b, N);
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- Benchmarking Fine-Grained ---
    int blocks_fine = (N + threads - 1) / threads;
    float time_fine;
    
    cudaEventRecord(start);
    fine_grained_kernel<<<blocks_fine, threads>>>(d_a, d_b, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_fine, start, stop);
    
    printf("Fine-Grained (1 iter/thread) Time: %f ms\n", time_fine);

    // --- Benchmarking Coarsened ---
    // Reduce number of blocks by COARSE_FACTOR
    int blocks_coarse = (N / COARSE_FACTOR + threads - 1) / threads;
    
    float time_coarse;
    cudaEventRecord(start);
    coarsened_kernel<<<blocks_coarse, threads>>>(d_a, d_b, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_coarse, start, stop);

    printf("Coarsened (%d iter/thread) Time: %f ms\n", COARSE_FACTOR, time_coarse);

    cudaFree(d_a);
    cudaFree(d_b);
    
    return 0;
}