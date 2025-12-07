#include <iostream>
#include <cuda_runtime.h>

__global__ void copyDataCoalesced(float *in, float *out, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        out[index] = in[index];
    }
}

void initializeArray(float *arr, int n) {
    for(int i = 0; i < n; ++i) {
        arr[i] = static_cast<float>(i);
    }
}

int main() {
    const int n = 1 << 24; // Adjust the data size for workload
    float *in, *out;

    cudaMallocManaged(&in, n * sizeof(float));
    cudaMallocManaged(&out, n * sizeof(float));

    initializeArray(in, n);

    int blockSize = 32; // Optimal block size for many devices
    int numBlocks = (n + blockSize - 1) / blockSize; // Calculate the number of blocks

    // Optimize grid dimensions based on device properties
    int minGridSize;
    
    // Calculates the best block size to maximize occupancy
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, copyDataCoalesced, 0, n);

    // Print suggested block size and minimum grid size
    std::cout << "Calculated Optimal Execution Parameters:" << std::endl;
    std::cout << "  Recommended Block Size: " << blockSize << std::endl;
    std::cout << "  Minimum Grid Size: " << minGridSize << std::endl;

    // Recalculate grid size based on the optimal block size
    numBlocks = (n + blockSize - 1) / blockSize;
    
    std::cout << "  Actual Grid Size used: " << numBlocks << std::endl;

    // Launch coalesced kernel with optimized parameters
    copyDataCoalesced<<<numBlocks, blockSize>>>(in, out, n);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
    
    cudaDeviceSynchronize();

    cudaFree(in);
    cudaFree(out);

    return 0;
}