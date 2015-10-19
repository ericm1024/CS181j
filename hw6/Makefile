CC = g++

CC_INCLUDE =
CC_FLAGS = -Wall -Wextra -std=c++11 -fopenmp
LD_FLAGS =

TARGETS = Main0 Main1 Main2

all: $(TARGETS)

Main0: Main0.cc
	$(CC) $< -o $@ -O3 $(CC_INCLUDE) $(LD_FLAGS) $(CC_FLAGS)

#feel free to use this target for your SIMD exploation
Main1: Main1.cc
	/cs/cs181j/common/gcc-5.2.0/bin/g++ $< -o $@ -O3 $(CC_INCLUDE) $(LD_FLAGS) $(CC_FLAGS) -fopt-info-vec-optimized=Main1.autovectorization.out -funsafe-math-optimizations -Wno-unused-but-set-variable -msse4 -mavx -fabi-version=6

Main2: Main2.cc Main2_functions.h
	$(CC) $< -o $@ -O3 $(CC_INCLUDE) $(LD_FLAGS) $(CC_FLAGS)

clean:
	rm -f $(TARGETS)

again: clean $(TARGETS)
