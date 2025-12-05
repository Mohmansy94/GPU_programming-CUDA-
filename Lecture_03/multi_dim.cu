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

// CUDA Kernel for Grayscale Conversion
// Converts RGB image to Grayscale using the formula: Y = 0.299*R + 0.587*G + 0.114*B
__global__ void grayscaleKernel(unsigned char* d_in, unsigned char* d_out, int width, int height, int channels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds
    // the image is stored in row-major order
    // so we need to check if the current thread is within the bounds of the image
    if (col < width && row < height) {
        int idx = (row * width + col) * channels;
        int out_idx = row * width + col;

        unsigned char r = d_in[idx];
        unsigned char g = d_in[idx + 1];
        unsigned char b = d_in[idx + 2];

        // Standard NTSC conversion formula
        d_out[out_idx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// CPU implementation for comparison
void grayscaleCPU(unsigned char* h_in, unsigned char* h_out, int width, int height, int channels) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * channels;
            int out_idx = y * width + x;

            unsigned char r = h_in[idx];
            unsigned char g = h_in[idx + 1];
            unsigned char b = h_in[idx + 2];

            h_out[out_idx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
        }
    }
}

int main() {
    const char* input_filename = "lena.png";
    const char* output_gpu_filename = "lena_gpu.png";
    const char* output_cpu_filename = "lena_cpu.png";

    int width, height, channels;
    
    // Load image
    unsigned char* h_img = stbi_load(input_filename, &width, &height, &channels, 0);
    if (!h_img) {
        std::cerr << "Error loading image: " << input_filename << std::endl;
        return 1;
    }

    std::cout << "Loaded " << input_filename << ": " << width << "x" << height << " channels: " << channels << std::endl;

    // We only support 3 (RGB) or 4 (RGBA) channels for this demo
    if (channels < 3) {
        std::cerr << "Image must have at least 3 channels." << std::endl;
        stbi_image_free(h_img);
        return 1;
    }

    size_t img_size = width * height * channels * sizeof(unsigned char);
    size_t out_size = width * height * sizeof(unsigned char); // Grayscale is 1 channel

    // Allocate host memory for outputs
    unsigned char* h_out_gpu = (unsigned char*)malloc(out_size);
    unsigned char* h_out_cpu = (unsigned char*)malloc(out_size);

    // Allocate device memory
    unsigned char *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, img_size));
    CHECK_CUDA(cudaMalloc(&d_out, out_size));

    // Copy input data to device
    CHECK_CUDA(cudaMemcpy(d_in, h_img, img_size, cudaMemcpyHostToDevice));

    // Define block and grid sizes
    dim3 numThreads(32, 32);
    dim3 numBlocks((width + numThreads.x - 1) / numThreads.x, (height + numThreads.y - 1) / numThreads.y);

    // --- GPU Execution ---
    std::cout << "Running on GPU..." << std::endl;
    
    // Warmup
    grayscaleKernel<<<numBlocks, numThreads>>>(d_in, d_out, width, height, channels);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Measure time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    grayscaleKernel<<<numBlocks, numThreads>>>(d_in, d_out, width, height, channels);
    cudaEventRecord(stop);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU Time: " << milliseconds << " ms" << std::endl;

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_out_gpu, d_out, out_size, cudaMemcpyDeviceToHost));

    // --- CPU Execution ---
    std::cout << "Running on CPU..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    grayscaleCPU(h_img, h_out_cpu, width, height, channels);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU Time: " << cpu_duration.count() << " ms" << std::endl;

    // --- Comparison ---
    std::cout << "Comparing results..." << std::endl;
    bool match = true;
    int diff_count = 0;
    for (int i = 0; i < width * height; ++i) {
        // Allow small difference due to floating point precision differences between CPU and GPU
        if (std::abs(h_out_gpu[i] - h_out_cpu[i]) > 1) {
            match = false;
            diff_count++;
            if (diff_count < 10) {
                 // std::cout << "Mismatch at index " << i << ": GPU=" << (int)h_out_gpu[i] << " CPU=" << (int)h_out_cpu[i] << std::endl;
            }
        }
    }

    if (match || diff_count == 0) { // Relaxed check
        std::cout << "Success! CPU and GPU results match." << std::endl;
    } else {
        std::cout << "Mismatch! Found " << diff_count << " differences." << std::endl;
        std::cout << "Note: Small differences (+/- 1) are expected due to floating point precision." << std::endl;
    }

    // Save images
    stbi_write_png(output_gpu_filename, width, height, 1, h_out_gpu, width);
    stbi_write_png(output_cpu_filename, width, height, 1, h_out_cpu, width);

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