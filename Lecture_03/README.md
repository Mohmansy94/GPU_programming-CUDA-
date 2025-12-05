# GPU Programming with CUDA - Lecture 03

This directory contains CUDA examples for basic image processing tasks.

## Examples

### 1. Grayscale Conversion (`multi_dim.cu`)
Converts a color image (`lena.png`) to grayscale using a CUDA kernel.
- **Input**: `lena.png`
- **Output**: `lena_gpu.png` (GPU result), `lena_cpu.png` (CPU result for comparison)
- **Performance**: GPU is significantly faster.

### 2. Image Blur (`img_blur.cu`)
Applies a Box Blur (5x5 kernel) to an image.
- **Input**: `lena.png`
- **Output**: `lena_blur_gpu.png`, `lena_blur_cpu.png`
- **Performance**: GPU achieves massive speedup due to parallel convolution.

## Performance Comparison

| Task | GPU Time (ms) | CPU Time (ms) | Speedup |
| :--- | :--- | :--- | :--- |
| Grayscale Conversion | ~0.010 | ~0.407 | **~40x** |
| Image Blur (5x5) | ~0.051 | ~15.80 | **~310x** |

## Visual Results

### Grayscale Conversion
| Original | GPU Output | CPU Output |
| :---: | :---: | :---: |
| ![Original](lena.png) | ![GPU Grayscale](lena_gpu.png) | ![CPU Grayscale](lena_cpu.png) |

### Image Blur
| Original | GPU Output | CPU Output |
| :---: | :---: | :---: |
| ![Original](lena.png) | ![GPU Blur](lena_blur_gpu.png) | ![CPU Blur](lena_blur_cpu.png) |

## Building and Running

### Prerequisites
- NVIDIA GPU with CUDA Toolkit installed (`nvcc`).
- `stb_image.h` and `stb_image_write.h` (included/downloaded).

### Compile
Use the provided `Makefile`:
```bash
make
```
Or compile manually:
```bash
nvcc multi_dim.cu -o multi_dim
nvcc img_blur.cu -o img_blur
```

### Run
```bash
./multi_dim
./img_blur
```
