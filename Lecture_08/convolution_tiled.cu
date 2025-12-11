
#include "config.h"

// Tiled GPU 2D Convolution Kernel using Shared Memory
// Uses a tile strategy where the Block corresponds to the OUTPUT tile.
// Threads load the Input tile (Block dim + halo) cooperatively.
__global__ void convolution_gpu_tiled(const float* input, float* output, const float* mask, int width, int height) {
    // Shared memory size is defined by the block size + halo
    // But since shared memory allocation in kernel launch must be 1D byte array, we cast it.
    // However, for simplicity with 2D indexing, we can use a slightly different approach or just linearization.
    // Let's assume dynamic shared memory: extern __shared__ float s_data[];
    extern __shared__ float s_input[];

    // Tile dimensions including halo
    int tile_w = blockDim.x + MASK_WIDTH - 1;
    int tile_h = blockDim.y + MASK_WIDTH - 1;

    // Global coordinates
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col_o = blockIdx.x * blockDim.x + tx;
    int row_o = blockIdx.y * blockDim.y + ty;

    // Linear thread ID within block
    int tid = ty * blockDim.x + tx;

    // Starting global coordinates for the block's input tile (top-left of halo)
    int row_i = (blockIdx.y * blockDim.y) - MASK_RADIUS;
    int col_i = (blockIdx.x * blockDim.x) - MASK_RADIUS;

    // Cooperative loading:
    // We need to load 'tile_w * tile_h' elements using 'blockDim.x * blockDim.y' threads.
    // Each thread might load multiple elements.
    
    int num_threads = blockDim.x * blockDim.y;
    int num_elements = tile_w * tile_h;

    for (int i = tid; i < num_elements; i += num_threads) {
        int cur_row = i / tile_w;
        int cur_col = i % tile_w;
        
        int global_r = row_i + cur_row;
        int global_c = col_i + cur_col;

        if (global_r >= 0 && global_r < height && global_c >= 0 && global_c < width) {
            s_input[cur_row * tile_w + cur_col] = input[global_r * width + global_c];
        } else {
            s_input[cur_row * tile_w + cur_col] = 0.0f;
        }
    }

    __syncthreads();

    // Compute Convolution
    if (col_o < width && row_o < height) {
        float sum = 0.0f;
        // In shared memory, the element corresponding to (tx, ty) starts at (ty + MASK_RADIUS, tx + MASK_RADIUS)
        // But we iterate simply:
        for (int i = 0; i < MASK_WIDTH; i++) {
            for (int j = 0; j < MASK_WIDTH; j++) {
                // The loaded tile has (0,0) at row_i, col_i.
                // The current pixel (row_o, col_o) is at offset (ty + radius, tx + radius) in shared mem.
                // We want input[row - radius + i].
                // Shared mem row index: ty + radius - radius + i = ty + i
                // Shared mem col index: tx + j
                sum += s_input[(ty + i) * tile_w + (tx + j)] * mask[i * MASK_WIDTH + j];
            }
        }
        output[row_o * width + col_o] = sum;
    }
}
