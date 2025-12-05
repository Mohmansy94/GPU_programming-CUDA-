NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_75

TARGETS = multi_dim img_blur

all: $(TARGETS)

multi_dim: multi_dim.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

img_blur: img_blur.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f $(TARGETS) *.o
