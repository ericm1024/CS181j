# CUDA code generation flags
GENCODE_FLAGS   := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35

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
  CUDA_PATH       ?= /usr/local/cuda-7.0
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

CC_FLAGS        += -O3 -std=c++11 -Wall -Wextra -fopenmp
NVCC_FLAGS      += -O3 -std=c++11

CC_INCLUDE      += -I$(CUDA_INC_PATH)

TARGETS = Cuda1DFDWaveInvasive Cuda1DFDWaveNonInvasive ManyMatrixMultiplications MatrixMultiplication

all: $(TARGETS)

Cuda1DFDWaveNonInvasive: Cuda1DFDWaveNonInvasive.cu
	$(NVCC) -ccbin=$(NVCCSCC) $< -o $@ $(CC_INCLUDE) $(NVCC_FLAGS) $(LD_FLAGS)

Cuda1DFDWaveInvasive: Cuda1DFDWaveInvasive.cu
	$(NVCC) -ccbin=$(NVCCSCC) $< -o $@ $(CC_INCLUDE) $(NVCC_FLAGS) $(LD_FLAGS)

ManyMatrixMultiplications_cuda.o: ManyMatrixMultiplications_cuda.cu ManyMatrixMultiplications_cuda.cuh
	$(NVCC) -ccbin=$(NVCCSCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) $(NVCC_INCLUDE) -o $@ -c $<

ManyMatrixMultiplications: ManyMatrixMultiplications.cc ManyMatrixMultiplications_cuda.o
	$(CC) $< -o $@ $(CC_INCLUDE) $(CC_FLAGS) ManyMatrixMultiplications_cuda.o $(LD_FLAGS)

MatrixMultiplication_cuda.o: MatrixMultiplication_cuda.cu MatrixMultiplication_cuda.cuh
	$(NVCC) -ccbin=$(NVCCSCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) $(NVCC_INCLUDE) -o $@ -c $<

MatrixMultiplication: MatrixMultiplication.cc MatrixMultiplication_cuda.o
	$(CC) $< -o $@ $(CC_INCLUDE) $(CC_FLAGS) MatrixMultiplication_cuda.o $(LD_FLAGS)

clean:
	rm -f *.o $(TARGETS)

again: clean $(TARGETS)
