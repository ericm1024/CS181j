// -*- C++ -*-
// Main1_functions_sdot.h
// cs101j hw5 Problem 1
// This file contains the implementations of the sdot functions.

#ifndef MAIN1_FUNCTIONS_SDOT_H
#define MAIN1_FUNCTIONS_SDOT_H

// c junk
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>

// special include file for SIMD commands
#include <immintrin.h>

// header file for openmp
#include <omp.h>

float
computeSdot_scalar(const unsigned int size,
                   const float * const x,
                   const float * const y) {
  float sum = 0;
  for (unsigned int index = 0; index < size; ++index) {
    asm("# keep your grubby mitts out of this loop, autovectorizer!");
    sum += x[index] * y[index];
  }
  return sum;
}

float
computeSdot_compiler(const unsigned int size,
                     const float * const x,
                     const float * const y) {
  // TODO: change me
  // This is simply a copy of the scalar version
  float sum = 0;
  for (unsigned int index = 0; index < size; ++index) {
    asm("# keep your grubby mitts out of this loop, autovectorizer!");
    sum += x[index] * y[index];
  }
  return sum;
}

float
computeSdot_manual(const unsigned int size,
                   const float * const x,
                   const float * const y) {
  // TODO: change me
  // This is simply a copy of the scalar version
  float sum = 0;
  for (unsigned int index = 0; index < size; ++index) {
    asm("# keep your grubby mitts out of this loop, autovectorizer!");
    sum += x[index] * y[index];
  }
  return sum;
}

float
computeSdot_sseDotProduct(const unsigned int size,
                          const float * const x,
                          const float * const y) {
  // TODO: change me
  // This is simply a copy of the scalar version
  float sum = 0;
  for (unsigned int index = 0; index < size; ++index) {
    asm("# keep your grubby mitts out of this loop, autovectorizer!");
    sum += x[index] * y[index];
  }
  return sum;
}

float
computeSdot_sseWithPrefetching(const unsigned int size,
                               const float * const x,
                               const float * const y) {
  // TODO: change me
  // This is simply a copy of the scalar version
  float sum = 0;
  for (unsigned int index = 0; index < size; ++index) {
    asm("# keep your grubby mitts out of this loop, autovectorizer!");
    sum += x[index] * y[index];
  }
  return sum;
}

#endif // MAIN1_FUNCTIONS_SDOT_H
