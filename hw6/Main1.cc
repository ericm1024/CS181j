// -*- C++ -*-
// Main1.cc
// cs181j hw6 SIMD exploration
// An example to illustrate how to implement simple SIMD vectorization on
//  something neat-o

// c junk
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>

// c++ junk
#include <array>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <string>

using std::string;
using std::vector;
using std::array;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

// These utilities are used on many assignments
#include "../Utilities.h"

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

int main() {

  const unsigned int N = 100;

  float * x = allocateAlignedMemory<float>(N, 64);
  freeAlignedMemory(&x);

  return 0;
}
