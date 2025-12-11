
#ifndef CONFIG_H
#define CONFIG_H

#include <vector>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

// Mask size (should be odd) - Using 7x7 for 2D
#define MASK_WIDTH 7
#define MASK_RADIUS (MASK_WIDTH / 2)

// Tile Dimension for Shared Memory (Output Tile)
#define TILE_WIDTH 16

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

// Function Declarations
// n is width/height (assuming square image)
void convolution_cpu(const float* input, float* output, const float* mask, int width, int height);

__global__ void convolution_gpu(const float* input, float* output, const float* mask, int width, int height);

void copy_mask_to_constant_memory(const float* host_mask);
__global__ void convolution_gpu_constant(const float* input, float* output, int width, int height);

__global__ void convolution_gpu_tiled(const float* input, float* output, const float* mask, int width, int height);

#endif // CONFIG_H
