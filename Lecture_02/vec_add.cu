#include<stdio.h>
#include<cuda_runtime.h>
#include<time.h>


#define SIZE 1024 * 1024* 32 // Define the vector size

// vector add function on the CPU

void vec_add_cpu(int *A, int *B, int *C, int n){

    for (int i = 0; i< n; i ++){
        C[i] = A[i] + B[i];
    } 
}

// Kernal 

__global__ void vec_add_gpu(int *A, int *B, int *C, int n){
    int i = threadIdx.x + blockDim.x * blockIdx.x; // This to idx the threads even in diffrent blocks

    // This to make sure we will not exceed the thread numbers
    if (i < n) {
        C[i] = A[i] + B[i];
    }

}


int main(){

    int *A, *B, *C; // host vector 
    int *d_A, *d_B, *d_C;  // device vector

    int size = SIZE * sizeof(int);

    // Allocate and initialize vector on CPU
    A = (int *)malloc(size);
    B = (int *)malloc(size);
    C = (int *)malloc(size);

    for (int i = 0; i < SIZE; i++){
        A[i] = i;
        B[i] = SIZE-1;
    }

    // Measure CPU time
    clock_t cpu_start = clock();
    vec_add_cpu(A, B, C, SIZE);
    clock_t cpu_end = clock();
    double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;
    printf("CPU Execution Time: %f seconds\n", cpu_time);

    // Allocate memory on GPU
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy input data form host to GPU (device)
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // CUDA event to measure the excution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start recording the event
    cudaEventRecord(start);

    // launch the Kernal and do the computations on the GPU
    int thredsPerBlock = 1024; 
    int blocksPerGrid = (SIZE + thredsPerBlock - 1) / thredsPerBlock;

    vec_add_gpu<<<blocksPerGrid, thredsPerBlock>>>(d_A, d_B, d_C, SIZE);

    // stop recording
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Execution Time: %f milliseconds\n", milliseconds);

    // copy the results from the GPU to the CPU
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Deallocate the memory on the device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}