// -*- C++ -*-
#ifndef COMMON_DEFINITIONS_H
#define COMMON_DEFINITIONS_H

// c junk
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdio>

// c++ junk
#include <array>
#include <vector>
#include <list>
#include <string>
#include <algorithm>
#include <chrono>
#include <random>
#include <fstream>

typedef std::array<float, 3> Point;

template <class T>
T *
allocateAlignedMemory(const unsigned int numberOfValues,
                      const unsigned int alignment) {
  return (T*)_mm_malloc(numberOfValues * sizeof(T), alignment);
}

template <class T>
void
freeAlignedMemory(T ** pointer) {
  _mm_free(*pointer);
  *pointer = 0;
}


#endif // COMMON_DEFINITIONS_H
