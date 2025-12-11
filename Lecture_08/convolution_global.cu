
#include "config.h"

// GPU 2D Convolution Kernel (Basic Global Memory)
__global__ void convolution_gpu(const float* input, float* output, const float* mask, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        float sum = 0.0f;
        for (int i = 0; i < MASK_WIDTH; i++) {
            for (int j = 0; j < MASK_WIDTH; j++) {
                int input_row = row - MASK_RADIUS + i;
                int input_col = col - MASK_RADIUS + j;

                if (input_row >= 0 && input_row < height && input_col >= 0 && input_col < width) {
                    sum += input[input_row * width + input_col] * mask[i * MASK_WIDTH + j];
                }
            }
        }
        output[row * width + col] = sum;
    }
}
