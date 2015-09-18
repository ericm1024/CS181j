CC_INCLUDE = 
LD_FLAGS = 
CC_FLAGS += -O3 -Wall -Wextra -std=c++11

TARGETS = Main3

all: $(TARGETS)

Main3: Main3.cc Main3_functors.h
	g++ $< -o $@ $(CC_INCLUDE) $(LD_FLAGS) $(CC_FLAGS)

clean:
	rm -f $(TARGETS)

again: clean $(TARGETS)
