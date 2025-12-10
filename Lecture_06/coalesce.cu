#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// Size of the array (32 MB * 4 bytes = 128 MB)
#define N (1 << 25)
#define BLOCK_SIZE 256

// Kernel to warm up the GPU/Memory
__global__ void warmup_kernel(float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        b[i] = a[i];
    }
}

// Stride 1 access (Coalesced)
__global__ void stride1_kernel(float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        b[i] = a[i] * 2.0f;
    }
}

// Stride 32 access (Uncoalesced)
// We process 'n' elements, but scattered in memory.
// To support this without massive memory, we will just use a large enough array 
// and only process a subset of it?
// Strategy:
// We want to process M elements.
// Stride 1: indexes 0, 1, ..., M-1
// Stride 32: indexes 0, 32, 64, ..., (M-1)*32
// This requires the array to be of size M*32.
// If M = 1<<20 (1 Million), M*32 = 32 Million (128 MB).
// So we can use N = 1<<25 (33 Million) and process M = N/32 elements safely.
__global__ void stride32_kernel(float* a, float* b, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        int idx = i * 32;
        b[idx] = a[idx] * 2.0f;
    }
}

int main() {
    float *d_a, *d_b;
    size_t size = N * sizeof(float); // Total memory size

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    
    // Initialize data (on device for speed, we don't care about values)
    cudaMemset(d_a, 0, size);

    int threads = BLOCK_SIZE;
    
    // Warmup
    int blocks_warmup = (N + threads - 1) / threads;
    warmup_kernel<<<blocks_warmup, threads>>>(d_a, d_b, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- Benchmark Coalesced (Stride 1) ---
    // We process M elements. Let M = N/32 to compare fair operation count?
    // No, compare Bandwidth.
    // If we process M elements with stride 1, we read M*4 bytes.
    // If we process M elements with stride 32, we read M*4 bytes (but transfer 32x more due to cache lines?)
    // Yes.
    // Let's use M = N/32 to run stride 32 safely.
    int M = N / 32;
    int blocks_M = (M + threads - 1) / threads;

    float time_stride1;
    cudaEventRecord(start);
    stride1_kernel<<<blocks_M, threads>>>(d_a, d_b, M);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_stride1, start, stop);

    printf("Stride 1 (Coalesced) Time (processing %d elements): %f ms\n", M, time_stride1);

    // --- Benchmark Uncoalesced (Stride 32) ---
    float time_stride32;
    cudaEventRecord(start);
    stride32_kernel<<<blocks_M, threads>>>(d_a, d_b, M);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_stride32, start, stop);

    printf("Stride 32 (Uncoalesced) Time (processing %d elements): %f ms\n", M, time_stride32);
    
    printf("Speedup (Coalesced vs Uncoalesced): %fx\n", time_stride32 / time_stride1);

    cudaFree(d_a);
    cudaFree(d_b);
    
    return 0;
}