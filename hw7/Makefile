CC = g++
#CC=/research/jeff/compilers/llvm_May2016/bin/clang++
CC_INCLUDE =
CC_FLAGS = -Wall -Wextra -std=c++1y -fopenmp
LD_FLAGS =

TARGETS = MaximalIndependentSet KMeansClustering FindIndexOfClosestPoint

all: $(TARGETS)

MaximalIndependentSet: MaximalIndependentSet.cc MaximalIndependentSet_functions.h GraphUtilities.h Graphs.h
	$(CC) $< -o $@ -O3 $(CC_INCLUDE) $(LD_FLAGS) $(CC_FLAGS)

KMeansClustering: KMeansClustering.cc KMeansClustering_functors.h
	$(CC) $< -o $@ -O3 $(CC_INCLUDE) $(LD_FLAGS) $(CC_FLAGS)

FindIndexOfClosestPoint: FindIndexOfClosestPoint.cc FindIndexOfClosestPoint_functions.h
	$(CC) $< -o $@ -O3 $(CC_INCLUDE) $(LD_FLAGS) $(CC_FLAGS)

clean:
	rm -f $(TARGETS)

again: clean $(TARGETS)
