
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// Error handling macro
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// 1. Naive Matrix Multiplication (Corrected)
__global__ void mat_mul_without_tiling(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 2. Tiled Matrix Multiplication (Square)
__global__ void mat_mul_tiled_square(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    // Iterate over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load data into shared memory
        int tiledRow = t * TILE_SIZE + threadIdx.y;
        int tiledCol = t * TILE_SIZE + threadIdx.x;

        if (row < N && tiledCol < N) {
            s_A[threadIdx.y][threadIdx.x] = A[row * N + tiledCol];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && tiledRow < N) {
            s_B[threadIdx.y][threadIdx.x] = B[tiledRow * N + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum for this tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// 3. Tiled Matrix Multiplication (General: MxK * KxN)
__global__ void mat_mul_tiled_general(float *A, float *B, float *C, int M, int K, int N) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load A tile
        int aRow = row;
        int aCol = t * TILE_SIZE + threadIdx.x;
        if (aRow < M && aCol < K) {
            s_A[threadIdx.y][threadIdx.x] = A[aRow * K + aCol];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile
        int bRow = t * TILE_SIZE + threadIdx.y;
        int bCol = col;
        if (bRow < K && bCol < N) {
            s_B[threadIdx.y][threadIdx.x] = B[bRow * N + bCol];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}


// 4. Tiled Matrix Multiplication (Optimized: No Divergence)
// Assumes N is a multiple of TILE_SIZE
__global__ void mat_mul_tiled_optimized(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    // Loop over tiles
    // Since N is multiple of TILE_SIZE, we don't need ceil/floor logic, just N/TILE_SIZE
    for (int t = 0; t < N / TILE_SIZE; ++t) {
        
        // Load data directly into shared memory without boundary checks
        // We assume valid memory access because N % TILE_SIZE == 0
        int tiledRow = t * TILE_SIZE + threadIdx.y;
        int tiledCol = t * TILE_SIZE + threadIdx.x;
        
        s_A[threadIdx.y][threadIdx.x] = A[row * N + tiledCol];
        s_B[threadIdx.y][threadIdx.x] = B[tiledRow * N + col];

        __syncthreads();

        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads();
    }
    
    // Write result
    C[row * N + col] = sum;
}

// CPU Verification
void mat_mul_cpu(float *A, float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

bool verify_result(float *C_gpu, float *C_cpu, int size) {
    for (int i = 0; i < size; i++) {
        if (fabs(C_gpu[i] - C_cpu[i]) > 1e-3) {
            std::cout << "Mismatch at index " << i << ": GPU " << C_gpu[i] << ", CPU " << C_cpu[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    int N = 1024; // Square matrix size
    int size = N * N * sizeof(float);

    // Host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    float *h_C_ref = (float*)malloc(size);

    // Initialize matrices
    srand(time(0));
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Compute CPU reference
    std::cout << "Computing CPU reference..." << std::endl;
    // For N=1024, CPU might be slow (approx 1024^3 ops ~ 1e9). Let's keep it but warn.
    mat_mul_cpu(h_A, h_B, h_C_ref, N, N, N);
    std::cout << "CPU reference computed." << std::endl;

    // Device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, size));

    // Copy to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Define Grid and Block dimensions
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // Create CUDA Events for timing
    cudaEvent_t start, stop;
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // --- Naive MatMul ---
    std::cout << "Running Naive MatMul..." << std::endl;
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    mat_mul_without_tiling<<<grid, block>>>(d_A, d_B, d_C, N);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Naive MatMul Time: " << milliseconds << " ms" << std::endl;
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    if (verify_result(h_C, h_C_ref, N * N)) std::cout << "Naive MatMul PASSED" << std::endl;
    else std::cout << "Naive MatMul FAILED" << std::endl;

    // --- Tiled MatMul (Square) ---
    std::cout << "Running Tiled MatMul (Square)..." << std::endl;
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    mat_mul_tiled_square<<<grid, block>>>(d_A, d_B, d_C, N);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Tiled MatMul (Square) Time: " << milliseconds << " ms" << std::endl;

    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    if (verify_result(h_C, h_C_ref, N * N)) std::cout << "Tiled MatMul (Square) PASSED" << std::endl;
    else std::cout << "Tiled MatMul (Square) FAILED" << std::endl;

    // --- Tiled MatMul (General) ---
    std::cout << "Running Tiled MatMul (General)..." << std::endl;
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    mat_mul_tiled_general<<<grid, block>>>(d_A, d_B, d_C, N, N, N);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Tiled MatMul (General) Time: " << milliseconds << " ms" << std::endl;

    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    if (verify_result(h_C, h_C_ref, N * N)) std::cout << "Tiled MatMul (General) PASSED" << std::endl;
    else std::cout << "Tiled MatMul (General) FAILED" << std::endl;

    // --- Tiled MatMul (Optimized) ---
    std::cout << "Running Tiled MatMul (Optimized)..." << std::endl;
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    mat_mul_tiled_optimized<<<grid, block>>>(d_A, d_B, d_C, N);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Tiled MatMul (Optimized) Time: " << milliseconds << " ms" << std::endl;

    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    if (verify_result(h_C, h_C_ref, N * N)) std::cout << "Tiled MatMul (Optimized) PASSED" << std::endl;
    else std::cout << "Tiled MatMul (Optimized) FAILED" << std::endl;

    // Free memory
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
