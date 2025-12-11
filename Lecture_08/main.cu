
#include "config.h"

int main() {
    int width = 1 << 10; // 1024
    int height = 1 << 10; // 1024
    int n = width * height; // 1M elements
    
    size_t bytes = n * sizeof(float);
    size_t mask_bytes = MASK_WIDTH * MASK_WIDTH * sizeof(float);

    printf("Image Size: %d x %d\n", width, height);
    printf("Mask Size: %d x %d\n", MASK_WIDTH, MASK_WIDTH);

    // Host memory
    std::vector<float> h_input(n);
    std::vector<float> h_output_cpu(n);
    std::vector<float> h_output_gpu(n);
    std::vector<float> h_output_gpu_constant(n);
    std::vector<float> h_output_gpu_tiled(n);
    std::vector<float> h_mask(MASK_WIDTH * MASK_WIDTH);

    // Initialize data
    srand(time(NULL));
    for (int i = 0; i < n; i++) h_input[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < MASK_WIDTH * MASK_WIDTH; i++) h_mask[i] = (float)(rand() % 10) / 10.0f;

    // --- CPU Computation ---
    printf("Running CPU Convolution...\n");
    auto start_cpu = std::chrono::high_resolution_clock::now();
    convolution_cpu(h_input.data(), h_output_cpu.data(), h_mask.data(), width, height);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;
    printf("CPU Time: %.3f ms\n", cpu_duration.count());

    // --- GPU Computation ---
    float *d_input, *d_output, *d_mask;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMalloc(&d_mask, mask_bytes));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mask, h_mask.data(), mask_bytes, cudaMemcpyHostToDevice));

    // Initialize Constant Memory
    copy_mask_to_constant_memory(h_mask.data());

    // Grid Dimensions for 2D
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    printf("Grid Size: (%d, %d), Block Size: (%d, %d)\n", gridSize.x, gridSize.y, blockSize.x, blockSize.y);

    // Warmup
    convolution_gpu<<<gridSize, blockSize>>>(d_input, d_output, d_mask, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 1. Measure Basic GPU (Global Memory)
    cudaEventRecord(start);
    convolution_gpu<<<gridSize, blockSize>>>(d_input, d_output, d_mask, width, height);
    cudaEventRecord(stop);
    CUDA_CHECK(cudaEventSynchronize(stop));
    float gpu_milliseconds = 0;
    cudaEventElapsedTime(&gpu_milliseconds, start, stop);
    
    printf("Basic GPU Time: %.3f ms\n", gpu_milliseconds);
    printf("Basic Speedup: %.2fx\n", cpu_duration.count() / gpu_milliseconds);

    // Verify Basic GPU
    CUDA_CHECK(cudaMemcpy(h_output_gpu.data(), d_output, bytes, cudaMemcpyDeviceToHost));
    bool match_basic = true;
    for (int i = 0; i < n; i++) {
        if (fabs(h_output_cpu[i] - h_output_gpu[i]) > 1e-3) {
            printf("Basic Mismatch at %d: CPU %f, GPU %f\n", i, h_output_cpu[i], h_output_gpu[i]);
            match_basic = false;
            break;
        }
    }
    if (match_basic) printf("Basic GPU PASSED\n");
    else printf("Basic GPU FAILED\n");


    // 2. Measure Constant Memory GPU
    cudaEventRecord(start);
    convolution_gpu_constant<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaEventRecord(stop);
    CUDA_CHECK(cudaEventSynchronize(stop));
    float gpu_const_milliseconds = 0;
    cudaEventElapsedTime(&gpu_const_milliseconds, start, stop);

    printf("Constant GPU Time: %.3f ms\n", gpu_const_milliseconds);
    printf("Constant Speedup: %.2fx\n", cpu_duration.count() / gpu_const_milliseconds);

    // Verify Constant Memory GPU
    CUDA_CHECK(cudaMemcpy(h_output_gpu_constant.data(), d_output, bytes, cudaMemcpyDeviceToHost));
    bool match_const = true;
    for (int i = 0; i < n; i++) {
        if (fabs(h_output_cpu[i] - h_output_gpu_constant[i]) > 1e-3) {
            if (match_const) printf("Constant Mismatch at %d: CPU %f, Constant %f\n", i, h_output_cpu[i], h_output_gpu_constant[i]);
            match_const = false;
            break;
        }
    }
    if (match_const) printf("Constant GPU PASSED\n");
    else printf("Constant GPU FAILED\n");


    // 3. Measure Tiled GPU
    // Shared memory size calculation:
    // TILE_WIDTH threads load a tile of (TILE_WIDTH + MASK_WIDTH - 1)^2 elements
    // We pass the size in bytes.
    int tile_dim_shared = TILE_WIDTH + MASK_WIDTH - 1;
    size_t shared_mem_size = tile_dim_shared * tile_dim_shared * sizeof(float);
    
    cudaEventRecord(start);
    convolution_gpu_tiled<<<gridSize, blockSize, shared_mem_size>>>(d_input, d_output, d_mask, width, height);
    cudaEventRecord(stop);
    CUDA_CHECK(cudaEventSynchronize(stop));
    float gpu_tiled_milliseconds = 0;
    cudaEventElapsedTime(&gpu_tiled_milliseconds, start, stop);

    printf("Tiled GPU Time: %.3f ms\n", gpu_tiled_milliseconds);
    printf("Tiled Speedup: %.2fx\n", cpu_duration.count() / gpu_tiled_milliseconds);

    // Verify Tiled GPU
    CUDA_CHECK(cudaMemcpy(h_output_gpu_tiled.data(), d_output, bytes, cudaMemcpyDeviceToHost));
    bool match_tiled = true;
    for (int i = 0; i < n; i++) {
        if (fabs(h_output_cpu[i] - h_output_gpu_tiled[i]) > 1e-3) {
             if (match_tiled) printf("Tiled Mismatch at %d: CPU %f, Tiled %f\n", i, h_output_cpu[i], h_output_gpu_tiled[i]);
            match_tiled = false;
            break;
        }
    }
    if (match_tiled) printf("Tiled GPU PASSED\n");
    else printf("Tiled GPU FAILED\n");


    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mask);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
