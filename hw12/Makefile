CC = mpic++

CC_FLAGS = -std=c++11 -Wall -Wextra

TARGETS = Main0 Integration KMeansClustering FrodosChocolates Funball Parallel1DFDWave

LD_FLAGS =

all: $(TARGETS)

Main0: Main0.cc
	$(CC) $< -o $@ -O3 $(CC_FLAGS) $(CC_INCLUDE) $(LD_FLAGS)

KMeansClustering: KMeansClustering.cc KMeansClustering_functors.h
	$(CC) $< -o $@ -O3 $(CC_FLAGS) $(CC_INCLUDE) $(LD_FLAGS)

Integration: Integration.cc
	$(CC) $< -o $@ -O3 $(CC_FLAGS) $(CC_INCLUDE) $(LD_FLAGS)

FrodosChocolates: FrodosChocolates.cc
	$(CC) $< -o $@ -O3 $(CC_FLAGS) $(CC_INCLUDE) $(LD_FLAGS)

Funball: Funball.cc FunballAnswerChecker.o
	$(CC) $< -o $@ -O3 $(CC_FLAGS) $(CC_INCLUDE) $(LD_FLAGS) FunballAnswerChecker.o

Parallel1DFDWave: Parallel1DFDWave.cc
	$(CC) $< -o $@ -O3 $(CC_FLAGS) $(CC_INCLUDE) $(LD_FLAGS)

clean:
	rm -f $(TARGETS)

again: clean $(TARGETS)
