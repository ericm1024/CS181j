// -*- C++ -*-
// Main1_functions_taylorExponential.h
// cs101j hw5 Problem 1
// This file contains the implementations of the taylor exponential functions.

#ifndef MAIN1_FUNCTIONS_TAYLOREXPONENTIAL_H
#define MAIN1_FUNCTIONS_TAYLOREXPONENTIAL_H

// c junk
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>

// special include file for SIMD commands
#include <immintrin.h>

void
computeTaylorExponential_scalar(const unsigned int size,
                                const float * const x,
                                const unsigned int numberOfTermsInExponential,
                                float * const y) {
  for (unsigned int index = 0; index < size; ++index) {
    asm("# keep your grubby mitts out of this loop, autovectorizer!");
    float exponential = 1;
    float power = 1;
    float factorial = 1;
    const float xValue = x[index];
    for (unsigned int powerIndex = 1; powerIndex < numberOfTermsInExponential;
         ++powerIndex) {
      power *= xValue;
      factorial *= powerIndex;
      exponential += power / factorial;
    }
    y[index] = exponential;
  }
}

void
computeTaylorExponential_compiler(const unsigned int size,
                                  const float * const x,
                                  const unsigned int numberOfTermsInExponential,
                                  float * const y) {
  // TODO: change me
  // This is simply a copy of the scalar version
  for (unsigned int index = 0; index < size; ++index) {
    asm("# keep your grubby mitts out of this loop, autovectorizer!");
    float exponential = 1;
    float power = 1;
    float factorial = 1;
    const float xValue = x[index];
    for (unsigned int powerIndex = 1; powerIndex < numberOfTermsInExponential;
         ++powerIndex) {
      power *= xValue;
      factorial *= powerIndex;
      exponential += power / factorial;
    }
    y[index] = exponential;
  }
}

void
computeTaylorExponential_manual(const unsigned int size,
                                const float * const x,
                                const unsigned int numberOfTermsInExponential,
                                float * const y) {
  // TODO: change me
  // This is simply a copy of the scalar version
  for (unsigned int index = 0; index < size; ++index) {
    asm("# keep your grubby mitts out of this loop, autovectorizer!");
    float exponential = 1;
    float power = 1;
    float factorial = 1;
    const float xValue = x[index];
    for (unsigned int powerIndex = 1; powerIndex < numberOfTermsInExponential;
         ++powerIndex) {
      power *= xValue;
      factorial *= powerIndex;
      exponential += power / factorial;
    }
    y[index] = exponential;
  }
}

#endif // MAIN1_FUNCTIONS_TAYLOREXPONENTIAL_H
