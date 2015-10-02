CC_INCLUDE = 
LD_FLAGS = 
CC_FLAGS += -O3 -Wall -Wextra -std=c++11
CC = g++

TARGETS = Main1 Main2 Main3

all: $(TARGETS)

Main1: Main1.cc Main1_functors.h
	$(CC) $< -o $@ $(CC_INCLUDE) $(LD_FLAGS) $(CC_FLAGS)

malloc_count.o:
	$(CC) malloc_count-0.7/malloc_count.c -o malloc_count.o -c

Main2: Main2.cc malloc_count.o
	$(CC) $< -o $@ -O3 $(CC_INCLUDE) $(LD_FLAGS) -Wall -std=c++11 malloc_count.o -ldl

Main3: Main3.cc
	$(CC) $< -o $@ $(CC_INCLUDE) $(LD_FLAGS) $(CC_FLAGS)

clean:
	rm -f $(TARGETS)

again: clean $(TARGETS)
