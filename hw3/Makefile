CXX=/research/jeff/compilers/llvm_May2016/bin/clang++
#CXX=g++
CC_INCLUDE = -I/cs/cs181j/common/papi-5.4.0-knuth/include
LD_FLAGS = -L/cs/cs181j/common/papi-5.4.0-knuth/lib -lpapi
CC_FLAGS += -O3 -Wall -Wextra -std=c++14

TARGETS = Main1

all: $(TARGETS)

Main1: Main1.cc Main1_functors.h
	$(CXX) $< -o $@ $(CC_INCLUDE) $(LD_FLAGS) $(CC_FLAGS)

clean:
	rm -f $(TARGETS)

again: clean $(TARGETS)
