
#include "config.h"

// Define Constant Memory for 2D Mask
__constant__ float d_mask_c[MASK_WIDTH * MASK_WIDTH];

// Helper to copy data to constant memory
void copy_mask_to_constant_memory(const float* host_mask) {
    CUDA_CHECK(cudaMemcpyToSymbol(d_mask_c, host_mask, MASK_WIDTH * MASK_WIDTH * sizeof(float)));
}

// GPU 2D Convolution Kernel (Constant Memory)
__global__ void convolution_gpu_constant(const float* input, float* output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        float sum = 0.0f;
        for (int i = 0; i < MASK_WIDTH; i++) {
            for (int j = 0; j < MASK_WIDTH; j++) {
                int input_row = row - MASK_RADIUS + i;
                int input_col = col - MASK_RADIUS + j;

                if (input_row >= 0 && input_row < height && input_col >= 0 && input_col < width) {
                    sum += input[input_row * width + input_col] * d_mask_c[i * MASK_WIDTH + j];
                }
            }
        }
        output[row * width + col] = sum;
    }
}
