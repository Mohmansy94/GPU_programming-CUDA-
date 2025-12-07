#include <iostream>
#include <cuda_runtime.h>
#include <vector>

// Dummy kernel for occupancy calculation
__global__ void genericKernel(int *a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = a[idx] * 2;
    }
}

int main() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    if (err != cudaSuccess) {
        std::cerr << "Error getting device count: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s)." << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    for (int i = 0; i < deviceCount; ++i) {
        cudaSetDevice(i);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Shared Memory per Block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
        std::cout << "  Registers per Block: " << prop.regsPerBlock << std::endl;
        std::cout << "  MultiProcessor Count: " << prop.multiProcessorCount << std::endl;

        int blockSize;   // The launch configurator returned block size 
        int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch 

        // Calculate occupancy
        // Note: We use 0 dynamic shared memory and 0 for logic-specific limit
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, genericKernel, 0, 0);

        std::cout << "  [Occupancy Calculator Result]" << std::endl;
        std::cout << "  Suggested Block Size: " << blockSize << std::endl;
        std::cout << "  Min Grid Size for Max Occupancy: " << minGridSize << std::endl;
        
        // Calculate theoretical occupancy
        int maxActiveBlocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, genericKernel, blockSize, 0);
        
        float occupancy = (maxActiveBlocks * blockSize / (float)prop.maxThreadsPerMultiProcessor) * 100;
        
        std::cout << "  Theoretical Max Active Blocks per SM: " << maxActiveBlocks << std::endl;
        std::cout << "  Theoretical Occupancy: " << occupancy << "%" << std::endl; 
        std::cout << "------------------------------------------------" << std::endl;
    }

    return 0;
}
