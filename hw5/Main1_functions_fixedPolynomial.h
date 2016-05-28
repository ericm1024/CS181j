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
#include <iostream>

// special include file for SIMD commands
#include <immintrin.h>

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
                                const float * __restrict__ const x,
                                const float c0,
                                const float c1,
                                const float c2,
                                const float c3,
                                float * __restrict__ const y) {

  for (unsigned int index = 0; index < size; ++index) {
    const float value = x[index];
    y[index] = c0 + c1 * value + c2 * value * value + c3 * value * value * value;
  }
}

void
computeFixedPolynomial_manual(const unsigned int size,
                              const float * __restrict__ const x,
                              const float _c0,
                              const float _c1,
                              const float _c2,
                              const float _c3,
                              float * __restrict__ const y) {
        
        auto i = 0u;

        __m256 c0 = _mm256_set1_ps(_c0);
        __m256 c1 = _mm256_set1_ps(_c1);
        __m256 c2 = _mm256_set1_ps(_c2);
        __m256 c3 = _mm256_set1_ps(_c3);


        const auto do_poly = [=](const __m256 val) {
                return c0 + c1*val + c2*val*val + c3*val*val*val;
        };
        
        for (; i < (size & ~31); i += 32) {
                __m256 val0 = _mm256_load_ps(x + i);
                __m256 val1 = _mm256_load_ps(x + i + 8);
                __m256 val2 = _mm256_load_ps(x + i + 16);
                __m256 val3 = _mm256_load_ps(x + i + 24);

                _mm256_store_ps(y+i, do_poly(val0));
                _mm256_store_ps(y+i+8, do_poly(val1));
                _mm256_store_ps(y+i+16, do_poly(val2));
                _mm256_store_ps(y+i+24, do_poly(val3));
        }
        
        for (; i < size; ++i) {
                const float value = x[i];
                y[i] = _c0 + _c1 * value + _c2 * value * value + _c3 * value * value * value;
        }
}

#endif // MAIN1_FUNCTIONS_FIXEDPOLYNOMIAL_H
