CC_INCLUDE = -I/cs/cs181j/common/tbb-4.3/include
CC_FLAGS = -O3 -Wall -std=c++11 -Wextra
LD_FLAGS = -L/cs/cs181j/common/tbb-4.3/build/linux_intel64_gcc_cc4.8_libc2.19_kernel3.13.0_release -ltbb

# CUDA code generation flags
GENCODE_FLAGS   := -gencode arch=compute_35,code=sm_35 -gencode arch=compute_30,code=sm_30

# OS-specific build flags
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  # compiler
  #CC              := /opt/local/bin/g++-mp-4.7
  #NVCCSCC         := /opt/local/bin/g++-mp-4.7
  CC              := g++
  NVCCSCC         := g++
  # cuda
  CUDA_PATH       ?= /usr/local/cuda
  CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib/
  LD_FLAGS        += -L$(CUDA_LIB_PATH) -lcudart -lcufft -lcublas
else
  # compiler
  CC              := g++
  NVCCSCC         := $(CC)
  # cuda
  CUDA_PATH       ?= /opt/cuda-7.5
  CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib64/
  ifeq ($(OS_SIZE),32)
    CC_FLAGS      += -m32
  else
    CC_FLAGS      += -m64
  endif
  LD_FLAGS      += -L$(CUDA_LIB_PATH) -lcudart -lcufft -lcublas

endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
  CC_FLAGS      += -m32
  NVCC_FLAGS    += -m32
else
  CC_FLAGS      += -m64
  NVCC_FLAGS    += -m64
endif

CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
NVCC            ?= $(CUDA_BIN_PATH)/nvcc

NVCC_FLAGS      += -O3
NVCC_FLAGS      += --std=c++11

TARGETS = Main0 Threaded1DFDWave Main2 Main3 Main4

all: $(TARGETS)

Threaded1DFDWave: Threaded1DFDWave.cc
	$(CC) $< -o $@ -O3 $(CC_INCLUDE) $(LD_FLAGS) $(CC_FLAGS) -fopenmp

Main2: Main2.cc
	$(CC) $< -o $@ -O3 $(CC_INCLUDE) $(LD_FLAGS) $(CC_FLAGS) -fopenmp

Main0: Main0.cc Main0_cuda.cuh Main0_cuda.o
	$(CC) $< -o $@ -O3 $(CC_INCLUDE) Main0_cuda.o $(LD_FLAGS) $(CC_FLAGS)

Main0_cuda.o: Main0_cuda.cu Main0_cuda.cuh
	$(NVCC) -ccbin=$(NVCCSCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) $(NVCC_INCLUDE) -o $@ -c $<

Main3: Main3.cc Main3_cuda.cuh Main3_cuda.o
	$(CC) $< -o $@ -O3 $(CC_INCLUDE) Main3_cuda.o $(LD_FLAGS) $(CC_FLAGS)

Main3_cuda.o: Main3_cuda.cu Main3_cuda.cuh
	$(NVCC) -ccbin=$(NVCCSCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) $(NVCC_INCLUDE) -o $@ -c $<

Main4: Main4.cc Main4_cuda.cuh Main4_cuda.o
	$(CC) $< -o $@ -O3 $(CC_INCLUDE) Main4_cuda.o $(LD_FLAGS) $(CC_FLAGS)

Main4_cuda.o: Main4_cuda.cu Main4_cuda.cuh
	$(NVCC) -ccbin=$(NVCCSCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) $(NVCC_INCLUDE) -o $@ -c $<

clean:
	rm -f $(TARGETS) *.o

again: clean $(TARGETS)
