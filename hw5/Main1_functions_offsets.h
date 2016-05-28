// -*- C++ -*-
// Main1_functions_offsets.h
// cs101j hw5 Problem 1
// This file contains the implementations of the offsets functions.

#ifndef MAIN1_FUNCTIONS_OFFSETS_H
#define MAIN1_FUNCTIONS_OFFSETS_H

// c junk
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>

// special include file for SIMD commands
#include <immintrin.h>

void
computeOffsets_scalar(const unsigned int size,
                      const float a,
                      const float b,
                      const float * const x,
                      const float * const y,
                      const float * const z,
                      float * const w) {
  for (unsigned int i = 0; i < size; ++i) {
    // the autovectorizer is so scared of mod we don't even have to
    //  put in an assembly comment.
    w[i] = a * x[i] * y[(i + 16)%size] / (b * z[(i + size - 8) % size]);
  }
}

void
computeOffsets_scalarNoMod(const unsigned int size,
                           const float a,
                           const float b,
                           const float * const x,
                           const float * const y,
                           const float * const z,
                           float * const w) {
  // TODO: change me
  // This is simply a copy of the scalar version
  for (unsigned int i = 0; i < size; ++i) {
    // the autovectorizer is so scared of mod we don't even have to
    //  put in an assembly comment.
    w[i] = a * x[i] * y[(i + 16)%size] / (b * z[(i + size - 8) % size]);
  }
}

void
computeOffsets_compiler(const unsigned int size,
                        const float a,
                        const float b,
                        const float * const x,
                        const float * const y,
                        const float * const z,
                        float * const w) {
  // TODO: change me
  // This is simply a copy of the scalar version
  for (unsigned int i = 0; i < size; ++i) {
    // the autovectorizer is so scared of mod we don't even have to
    //  put in an assembly comment.
    w[i] = a * x[i] * y[(i + 16)%size] / (b * z[(i + size - 8) % size]);
  }
}

void
computeOffsets_manual(const unsigned int size,
                      const float a,
                      const float b,
                      const float * const x,
                      const float * const y,
                      const float * const z,
                      float * const w) {
  // TODO: change me
  // This is simply a copy of the scalar version
  for (unsigned int i = 0; i < size; ++i) {
    // the autovectorizer is so scared of mod we don't even have to
    //  put in an assembly comment.
    w[i] = a * x[i] * y[(i + 16)%size] / (b * z[(i + size - 8) % size]);
  }
}

#endif // MAIN1_FUNCTIONS_OFFSETS_H
