CC=/cs/cs181j/common/gcc-5.2.0/bin/g++
#CC=/research/jeff/compilers/llvm_May2016/bin/clang++
CC_INCLUDE =
CC_FLAGS = -Wall -Wextra -std=c++11 -funsafe-math-optimizations -Wno-unused-but-set-variable -msse4 -mavx -march=native -fabi-version=6 -fopenmp 
LD_FLAGS =

TARGETS = Main0 Main1 Main2 KMeansClustering

all: $(TARGETS)

Main0: Main0.cc
	$(CC) $< -o $@ -O3 $(CC_INCLUDE) $(LD_FLAGS) $(CC_FLAGS) -fopt-info-vec-optimized=autovectorization.out.$@

Main1: Main1.cc Main1_functions_sdot.h Main1_functions_fixedPolynomial.h Main1_functions_offsets.h Main1_functions_taylorExponential.h
	rm -f autovectorization.out.$@
	$(CC) $< -o $@ -O3 $(CC_INCLUDE) $(LD_FLAGS) $(CC_FLAGS) -fopt-info-vec-optimized=autovectorization.out.$@

data_p1: Main1
	./Main1
	python generatePlots1.py
	pdflatex hw5

Main2: Main2.cc Main2_functions_sqrt.h Main2_functions_fixedPolynomial.h
	rm -f autovectorization.out.$@
	$(CC) $< -o $@ -O3 $(CC_INCLUDE) $(LD_FLAGS) $(CC_FLAGS) -fopt-info-vec-optimized=autovectorization.out.$@

data_p2: Main2
	./Main2
	python generatePlots2.py
	pdflatex hw5

KMeansClustering: KMeansClustering.cc KMeansClustering_functors.h
	rm -f autovectorization.out.$@
	$(CC) $< -o $@ -O3 $(CC_INCLUDE) $(LD_FLAGS) $(CC_FLAGS) -fopt-info-vec-optimized=autovectorization.out.$@

clean:
	rm -f $(TARGETS) *.o

again: clean $(TARGETS)
