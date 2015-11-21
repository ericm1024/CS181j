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

CC_FLAGS        += -O3 -std=c++11 -Wall -Wextra
NVCC_FLAGS      += -O3

TARGETS = BinarySearch KMeansClustering MatrixMultiplication ReductionRace

all: $(TARGETS)

BinarySearch_cuda.o: BinarySearch_cuda.cu BinarySearch_cuda.cuh
	$(NVCC) -ccbin=$(NVCCSCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) $(NVCC_INCLUDE) -o $@ -c $<

BinarySearch: BinarySearch.cc BinarySearch_cuda.o
	$(CC) $< -o $@ $(CC_INCLUDE) $(CC_FLAGS) BinarySearch_cuda.o $(LD_FLAGS)

KMeansClustering_cuda.o: KMeansClustering_cuda.cu KMeansClustering_cuda.cuh
	$(NVCC) -ccbin=$(NVCCSCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) $(NVCC_INCLUDE) -o $@ -c $<

KMeansClustering: KMeansClustering.cc KMeansClustering_cuda.o
	$(CC) $< -o $@ $(CC_INCLUDE) $(CC_FLAGS) KMeansClustering_cuda.o $(LD_FLAGS)

MatrixMultiplication_cuda.o: MatrixMultiplication_cuda.cu MatrixMultiplication_cuda.cuh
	$(NVCC) -ccbin=$(NVCCSCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) $(NVCC_INCLUDE) -o $@ -c $<

MatrixMultiplication: MatrixMultiplication.cc MatrixMultiplication_cuda.o
	$(CC) $< -o $@ $(CC_INCLUDE) $(CC_FLAGS) MatrixMultiplication_cuda.o $(LD_FLAGS)

ReductionRace_cuda.o: ReductionRace_cuda.cu ReductionRace_cuda.cuh
	$(NVCC) -ccbin=$(NVCCSCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) $(NVCC_INCLUDE) -o $@ -c $<

ReductionRace: ReductionRace.cc ReductionRace_cuda.o
	$(CC) $< -o $@ $(CC_INCLUDE) $(CC_FLAGS) ReductionRace_cuda.o $(LD_FLAGS)

clean:
	rm -f *.o $(TARGETS)

again: clean $(TARGETS)
