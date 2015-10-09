// -*- C++ -*-
// Main1_functions_fixedPolynomial.h
// cs101j hw5 Problem 1
// This file contains the implementations of the fixedPolynomial functions.

#ifndef MAIN1_FUNCTIONS_FIXEDPOLYNOMIAL_H
#define MAIN1_FUNCTIONS_FIXEDPOLYNOMIAL_H

// c junk
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>

// special include file for SIMD commands
#include <immintrin.h>

// header file for openmp
#include <omp.h>

void
computeFixedPolynomial_scalar(const unsigned int size,
                              const float * const x,
                              const float c0,
                              const float c1,
                              const float c2,
                              const float c3,
                              float * const y) {
  for (unsigned int index = 0; index < size; ++index) {
    asm("# keep your grubby mitts out of this loop, autovectorizer!");
    const float value = x[index];
    y[index] = c0 + c1 * value + c2 * value * value + c3 * value * value * value;
  }
}

void
computeFixedPolynomial_compiler(const unsigned int size,
                                const float * const x,
                                const float c0,
                                const float c1,
                                const float c2,
                                const float c3,
                                float * const y) {
  // TODO: change me
  // This is simply a copy of the scalar version
  for (unsigned int index = 0; index < size; ++index) {
    asm("# keep your grubby mitts out of this loop, autovectorizer!");
    const float value = x[index];
    y[index] = c0 + c1 * value + c2 * value * value + c3 * value * value * value;
  }
}

void
computeFixedPolynomial_manual(const unsigned int size,
                              const float * const x,
                              const float c0,
                              const float c1,
                              const float c2,
                              const float c3,
                              float * const y) {
  // TODO: change me
  // This is simply a copy of the scalar version
  for (unsigned int index = 0; index < size; ++index) {
    asm("# keep your grubby mitts out of this loop, autovectorizer!");
    const float value = x[index];
    y[index] = c0 + c1 * value + c2 * value * value + c3 * value * value * value;
  }
}

#endif // MAIN1_FUNCTIONS_FIXEDPOLYNOMIAL_H
