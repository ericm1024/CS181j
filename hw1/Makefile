CC_INCLUDE = -I/cs/cs181j/common/papi-5.4.0-knuth/include
LD_FLAGS = -L/cs/cs181j/common/papi-5.4.0-knuth/lib -lpapi
CC_FLAGS += -O3 -Wall -Wextra -std=c++1y

TARGETS = Main0 Main2 Main3

all: $(TARGETS)

Main0: Main0.cc
	g++ $< -o $@ $(CC_INCLUDE) $(LD_FLAGS) $(CC_FLAGS)

Main2: Main2.cc
	g++ $< -o $@ $(CC_INCLUDE) $(LD_FLAGS) $(CC_FLAGS)

Main3: Main3.cc
	g++ $< -o $@ $(CC_INCLUDE) $(LD_FLAGS) $(CC_FLAGS)

clean:
	rm -f $(TARGETS)

again: clean $(TARGETS)
