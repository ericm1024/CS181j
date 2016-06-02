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
                                  const float * __restrict__ const x,
                                  const unsigned int numberOfTermsInExponential,
                                  float * __restrict__ const y) {
  // TODO: change me
  // This is simply a copy of the scalar version
  for (unsigned int index = 0; index < size; ++index) {
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
                                const float * __restrict__ const x,
                                const unsigned int nr_terms,
                                float * __restrict__ const y) {

        const auto stride = 8;
        auto i = 0u;
        for (; i < (size & ~(stride - 1)); i += stride) {
                float factorial = 1;
                __m256 exp = _mm256_set1_ps(1.);
                __m256 power = _mm256_set1_ps(1.);
                const __m256 x_vec = _mm256_load_ps(x + i);
                for (unsigned int powerIndex = 1; powerIndex < nr_terms; ++powerIndex) {
                        power *= x_vec;
                        factorial *= powerIndex;
                        exp += power * _mm256_rcp_ps(_mm256_set1_ps(factorial));
                }
                _mm256_store_ps(y+i, exp);
        }

        for (; i < size; ++i) {
                float exponential = 1;
                float power = 1;
                float factorial = 1;
                const float xValue = x[i];
                for (unsigned int powerIndex = 1; powerIndex < nr_terms; ++powerIndex) {
                        power *= xValue;
                        factorial *= powerIndex;
                        exponential += power / factorial;
                }
                y[i] = exponential;
        }
}

#endif // MAIN1_FUNCTIONS_TAYLOREXPONENTIAL_H
