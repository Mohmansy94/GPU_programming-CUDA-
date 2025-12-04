# How to Run `vec_add.cu`

This document explains how to compile and run the Vector Addition CUDA program.

## Prerequisites

- NVIDIA GPU
- CUDA Toolkit installed (specifically `nvcc` compiler)

## Compilation

To compile the code, use the `nvcc` compiler. Open your terminal and navigate to the directory containing `vec_add.cu`, then run:

```bash
nvcc vec_add.cu -o vec_add
```

This command compiles `vec_add.cu` and creates an executable named `vec_add`.

## Execution

To run the compiled program, execute the following command in your terminal:

```bash
./vec_add
```

## Expected Output

You should see output similar to the following, showing the execution time for both CPU and GPU implementations:

```text
CPU Execution Time: 0.090312 seconds
GPU Execution Time: 25.715008 milliseconds
```

*Note: Actual times will vary depending on your hardware.*
