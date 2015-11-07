CC = g++

CC_INCLUDE = -I/cs/cs181j/common/tbb-4.3/include
CC_FLAGS = -O3 -Wall -std=c++11 -fopenmp -Wextra
LD_FLAGS = -L/cs/cs181j/common/tbb-4.3/build/linux_intel64_gcc_cc4.8_libc2.19_kernel3.13.0_release -ltbb

TARGETS = Main0 FindIndexOfClosestPoint Main2 Main3

all: $(TARGETS)

Main0: Main0.cc
	$(CC) $< -o $@ $(CC_INCLUDE) $(LD_FLAGS) $(CC_FLAGS)

FindIndexOfClosestPoint: FindIndexOfClosestPoint.cc FindIndexOfClosestPoint_functions.h
	$(CC) $< -o $@ $(CC_INCLUDE) $(LD_FLAGS) $(CC_FLAGS)

Main2: Main2.cc Main2_functions.h
	$(CC) $< -o $@ $(CC_INCLUDE) $(LD_FLAGS) $(CC_FLAGS)

Main3: Main3.cc Main3_functions.h
	$(CC) $< -o $@ $(CC_INCLUDE) $(LD_FLAGS) $(CC_FLAGS)

clean:
	rm -f $(TARGETS)

again: clean $(TARGETS)
