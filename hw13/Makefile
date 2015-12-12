CC = mpic++

CC_FLAGS = -std=c++11 -Wall -Wextra

CC_INCLUDE = -I/cs/cs181j/common/stlib/src -I/cs/cs181j/common/boost_1_57_0

TARGETS = Mini2dMD_BigPicture Mini2dMD_NeedToKnow

LD_FLAGS =

all: $(TARGETS)

Mini2dMD_BigPicture: Mini2dMD_BigPicture.cc Mini2dMD.h
	$(CC) $< -o $@ -O3 $(CC_FLAGS) $(CC_INCLUDE) $(LD_FLAGS)

Mini2dMD_NeedToKnow: Mini2dMD_NeedToKnow.cc Mini2dMD.h
	$(CC) $< -o $@ -O3 $(CC_FLAGS) $(CC_INCLUDE) $(LD_FLAGS)

clean:
	rm -f *.o $(TARGETS)

again: clean $(TARGETS)
