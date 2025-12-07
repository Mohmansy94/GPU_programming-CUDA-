# GPU Programming with CUDA

This repository contains code examples, notes, and performance benchmarks for learning GPU Programming with CUDA. It follows the ["GPU Programming" video series](https://www.youtube.com/watch?v=4pkbXmE4POc&list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4) to explore GPU architecture and optimization techniques.

## Repository Structure
The project is organized by lectures, each containing source code (`.cu`), build scripts, and detailed documentation.

*   **Lecture 02:** Introduction to CUDA, Host/Device memory, and basic kernel launches.
*   **Lecture 03:** Grid and Block dimensions, Thread indexing.
*   **Lecture 04:** GPU Architecture (Ampere), Control Divergence, Occupancy, and Latency Hiding.
    *   Includes `divergence.cu` to demonstrate warp divergence performance impact.
    *   Includes `occupancy.cu` and `device_occupancy.cu` for calculating optimal launch configurations.

## Future Updates
Based on the course curriculum, upcoming lectures and code modules will cover:

1.  **Memory Optimization:**
    *   Global Memory Coalescing
    *   Shared Memory Tiling and Bank Conflicts
    *   Constant and Texture Memory usage
2.  **Parallel Patterns:**
    *   Parallel Reduction (Sum, Max, etc.)
    *   Prefix Sum (Scan) algorithms
    *   Histogram calculation
3.  **Concurrency & Streams:**
    *   Asynchronous memory transfers
    *   CUDA Streams for overlapping compute and copy
4.  **Libraries:**
    *   cuBLAS, cuFFT, and Thrust library usage.

## Prerequisites
*   NVIDIA GPU (Compute Capability 5.0+)
*   CUDA Toolkit (11.0+)
*   C++ Compiler (gcc/g++)