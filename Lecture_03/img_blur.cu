#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cuda_runtime.h>

// CUDA Error checking macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define BLUR_SIZE 5

// CUDA Kernel for Box Blur
__global__ void blurKernel(unsigned char* d_in, unsigned char* d_out, int width, int height, int channels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int pixVal[3] = {0, 0, 0}; // R, G, B accumulators
        int pixels = 0;

        // Iterate over the kernel window
        // the image is stored in row-major order
        // so we need to check if the current thread is within the bounds of the image
        for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                // Check bounds
                if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                    int curIdx = (curRow * width + curCol) * channels;
                    pixVal[0] += d_in[curIdx];
                    pixVal[1] += d_in[curIdx + 1];
                    pixVal[2] += d_in[curIdx + 2];
                    pixels++;
                }
            }
        }

        // Write averaged value
        int outIdx = (row * width + col) * channels;
        d_out[outIdx]     = (unsigned char)(pixVal[0] / pixels);
        d_out[outIdx + 1] = (unsigned char)(pixVal[1] / pixels);
        d_out[outIdx + 2] = (unsigned char)(pixVal[2] / pixels);
        if (channels > 3) { // Preserve alpha if present
             d_out[outIdx + 3] = d_in[outIdx + 3];
        }
    }
}

// CPU implementation for comparison
void blurCPU(unsigned char* h_in, unsigned char* h_out, int width, int height, int channels) {
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int pixVal[3] = {0, 0, 0};
            int pixels = 0;

            for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow) {
                for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol) {
                    int curRow = row + blurRow;
                    int curCol = col + blurCol;

                    if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                        int curIdx = (curRow * width + curCol) * channels;
                        pixVal[0] += h_in[curIdx];
                        pixVal[1] += h_in[curIdx + 1];
                        pixVal[2] += h_in[curIdx + 2];
                        pixels++;
                    }
                }
            }

            int outIdx = (row * width + col) * channels;
            h_out[outIdx]     = (unsigned char)(pixVal[0] / pixels);
            h_out[outIdx + 1] = (unsigned char)(pixVal[1] / pixels);
            h_out[outIdx + 2] = (unsigned char)(pixVal[2] / pixels);
            if (channels > 3) {
                h_out[outIdx + 3] = h_in[outIdx + 3];
            }
        }
    }
}

int main() {
    const char* input_filename = "lena.png";
    const char* output_gpu_filename = "lena_blur_gpu.png";
    const char* output_cpu_filename = "lena_blur_cpu.png";

    int width, height, channels;
    
    // Load image
    unsigned char* h_img = stbi_load(input_filename, &width, &height, &channels, 0);
    if (!h_img) {
        std::cerr << "Error loading image: " << input_filename << std::endl;
        return 1;
    }

    std::cout << "Loaded " << input_filename << ": " << width << "x" << height << " channels: " << channels << std::endl;

    if (channels < 3) {
        std::cerr << "Image must have at least 3 channels." << std::endl;
        stbi_image_free(h_img);
        return 1;
    }

    size_t img_size = width * height * channels * sizeof(unsigned char);

    // Allocate host memory for outputs
    unsigned char* h_out_gpu = (unsigned char*)malloc(img_size);
    unsigned char* h_out_cpu = (unsigned char*)malloc(img_size);

    // Allocate device memory
    unsigned char *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, img_size));
    CHECK_CUDA(cudaMalloc(&d_out, img_size));

    // Copy input data to device
    CHECK_CUDA(cudaMemcpy(d_in, h_img, img_size, cudaMemcpyHostToDevice));

    // Define block and grid sizes
    dim3 numThreads(16, 16);
    dim3 numBlocks((width + numThreads.x - 1) / numThreads.x, (height + numThreads.y - 1) / numThreads.y);

    // --- GPU Execution ---
    std::cout << "Running on GPU..." << std::endl;
    
    // Warmup
    blurKernel<<<numBlocks, numThreads>>>(d_in, d_out, width, height, channels);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Measure time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    blurKernel<<<numBlocks, numThreads>>>(d_in, d_out, width, height, channels);
    cudaEventRecord(stop);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU Time: " << milliseconds << " ms" << std::endl;

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_out_gpu, d_out, img_size, cudaMemcpyDeviceToHost));

    // --- CPU Execution ---
    std::cout << "Running on CPU..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    blurCPU(h_img, h_out_cpu, width, height, channels);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU Time: " << cpu_duration.count() << " ms" << std::endl;

    // --- Comparison ---
    std::cout << "Comparing results..." << std::endl;
    bool match = true;
    int diff_count = 0;
    for (int i = 0; i < width * height * channels; ++i) {
        if (std::abs(h_out_gpu[i] - h_out_cpu[i]) > 1) {
            match = false;
            diff_count++;
            if (diff_count < 10) {
                 // std::cout << "Mismatch at index " << i << ": GPU=" << (int)h_out_gpu[i] << " CPU=" << (int)h_out_cpu[i] << std::endl;
            }
        }
    }

    if (match || diff_count == 0) {
        std::cout << "Success! CPU and GPU results match." << std::endl;
    } else {
        std::cout << "Mismatch! Found " << diff_count << " differences." << std::endl;
    }

    // Save images
    stbi_write_png(output_gpu_filename, width, height, channels, h_out_gpu, width * channels);
    stbi_write_png(output_cpu_filename, width, height, channels, h_out_cpu, width * channels);

    std::cout << "Saved outputs to " << output_gpu_filename << " and " << output_cpu_filename << std::endl;

    // Cleanup
    stbi_image_free(h_img);
    free(h_out_gpu);
    free(h_out_cpu);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
