
NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_70

SRCS = main.cu convolution_cpu.cu convolution_global.cu convolution_constant.cu convolution_tiled.cu
OBJS = $(SRCS:.cu=.o)
TARGET = convolution

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

%.o: %.cu config.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean
