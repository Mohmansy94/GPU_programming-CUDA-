#include <stdio.h>
#include <cuda_runtime.h>
#include <assert.h>

/*
 * Kernel illustrating Warp Divergence.
 * 
 * In this kernel, threads in the same warp may take different execution paths
 * based on the condition (data[idx] % 2 == 0).
 * - Even numbers execute the 'if' block.
 * - Odd numbers execute the 'else' block.
 * 
 * Since threads in a warp execute in lock-step (SIMT - Single Instruction Multiple Threads),
 * when threads diverge, the hardware serializes the execution paths. 
 * The warp first executes threads taking the 'if' path (while disabling others),
 * and then executes threads taking the 'else' path. 
 * This effectively doubles the number of instruction cycles for the divergent section,
 * reducing performance.
 */
__global__ void divergenceKernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (data[idx] % 2 == 0) {
            data[idx] = data[idx] * 2;
        } else {
            data[idx] = data[idx] + 1;
        }
    }
}

/*
 * Kernel avoiding Warp Divergence using predication/math.
 * 
 * Here, we replace the conditional branching with arithmetic operations.
 * - isEven evaluates to 1 for even numbers, 0 for odd.
 * - (!isEven) evaluates to 0 for even numbers, 1 for odd.
 * 
 * Both parts of the expression are evaluated for all threads, but since there is no
 * control flow branching (no if/else), all threads in the warp execute the same 
 * instructions sequence. This maintains high utilization and avoids divergence serialization.
 * 
 */

__global__ void withoutDivergenceKernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int val = data[idx];
        int isEven = !(val % 2);
        // Calculation performed without branching
        data[idx] = isEven * (val * 2) + (!isEven) * (val + 1);
    }
}   

/*
 * Function to verify the results calculated by the GPU.
 * It computes the expected result on the CPU and compares it.
 */
void verify_results(int *data, int n) {
    int error_count = 0;
    for (int i = 0; i < n; i++) {
        int original = i; // Based on initialization logic data[i] = i
        int expected;
        if (original % 2 == 0) {
            expected = original * 2;
        } else {
            expected = original + 1;
        }
        
        if (data[i] != expected) {
            if (error_count < 5) {
                printf("Error at index %d: Expected %d, Got %d\n", i, expected, data[i]);
            }
            error_count++;
        }
    }
    if (error_count == 0) {
        printf("Verification Passed!\n");
    } else {
        printf("Verification Failed! Total errors: %d\n", error_count);
    }
}

/*
 * Helper function to measure kernel execution time and handle memory transfers.
 * Returns the execution time in milliseconds.
 */
float benchmarkKernel(void (*kernel)(int *, int), int *data, int N, const char *kernelName) {
    int *devData;
    // Allocate memory on GPU
    cudaMalloc(&devData, N * sizeof(int));
    // Copy input data from Host to Device
    cudaMemcpy(devData, data, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Record start event
    cudaEventRecord(start);
    
    // Launch Kernel
    kernel<<<blocksPerGrid, threadsPerBlock>>>(devData, N);
    
    // Record stop event
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%s took %f milliseconds\n", kernelName, milliseconds);

    // Copy results back to Host
    cudaMemcpy(data, devData, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(devData);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

int main() {
    // Increase size to make performance difference more noticeable and ensure multiple warps/blocks
    // 1 << 20 is approx 1 million elements
    int n = 1 << 20; 
    printf("Running benchmark with %d elements...\n", n);

    int *data = (int*)malloc(n * sizeof(int));
    
    // --- Test 1: Divergence Kernel ---
    // Initialize data
    for (int i = 0; i < n; i++) {
        data[i] = i;
    }   

    float time1 = benchmarkKernel(divergenceKernel, data, n, "divergenceKernel");
    verify_results(data, n);

    printf("------------------------------------------------\n");

    // --- Test 2: Without Divergence Kernel ---
    // Reset data for fair comparison
    for (int i = 0; i < n; i++) {
        data[i] = i;
    } 

    float time2 = benchmarkKernel(withoutDivergenceKernel, data, n, "withoutDivergenceKernel");
    verify_results(data, n);

    printf("------------------------------------------------\n");
    float improvement = (time1 - time2) / time1 * 100.0f;
    float speedup = time1 / time2;
    printf("Performance: \n");
    printf("  Speedup: %.2fx faster\n", speedup);
    printf("  Execution Time Reduction: %.2f%%\n", improvement);

    free(data);
    return 0;
}
