
#include "config.h"

// CPU 2D Convolution
void convolution_cpu(const float* input, float* output, const float* mask, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
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
}
