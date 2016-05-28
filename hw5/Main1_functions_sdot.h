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

inline float computeSdot_compiler(const unsigned int size,
                                  const float * __restrict__ const x,
                                               // ayy lmao
                                  const float * __restrict__ const y) {
  float sum = 0;
  for (unsigned int index = 0; index < size; ++index) {
    sum += x[index] * y[index];
  }
  return sum;
}

float
computeSdot_manual(const unsigned int size,
                   const float * __restrict__ const x,
                   const float * __restrict__ const y) {
        __m256 vsum1 = _mm256_setzero_ps();
        __m256 vsum2 = _mm256_setzero_ps();
        auto i = 0u;
        for (; i < (size & ~15); i += 16) {
                vsum1 += _mm256_load_ps(x + i) * _mm256_load_ps(y + i);
                vsum2 += _mm256_load_ps(x + i + 8) * _mm256_load_ps(y + i + 8);
        }

        const auto sum256_ps = [](const __m256 a) {
                return a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7];
        };
        
        float sum = sum256_ps(vsum1) + sum256_ps(vsum2);
        for (; i < size; ++i)
                sum += x[i] * y[i];

        return sum;
}

float
computeSdot_sseDotProduct(const unsigned int size,
                          const float * __restrict__ const x,
                          const float * __restrict__ const y) {
        __m256 vsum1 = _mm256_setzero_ps();
        __m256 vsum2 = _mm256_setzero_ps();
        __m256 vsum3 = _mm256_setzero_ps();
        __m256 vsum4 = _mm256_setzero_ps();
        const uint8_t mask = 0xff;
        auto i = 0u;
        for (; i < (size & ~31); i += 32) {
                vsum1 += _mm256_dp_ps(_mm256_load_ps(x + i),
                                      _mm256_load_ps(y + i),
                                      mask);
                vsum2 += _mm256_dp_ps(_mm256_load_ps(x + i + 8),
                                      _mm256_load_ps(y + i + 8),
                                      mask);
                vsum3 += _mm256_dp_ps(_mm256_load_ps(x + i + 16),
                                      _mm256_load_ps(y + i + 16),
                                      mask);
                vsum4 += _mm256_dp_ps(_mm256_load_ps(x + i + 24),
                                      _mm256_load_ps(y + i + 24),
                                      mask);
        }

        float sum = vsum1[0] + vsum1[4] + vsum2[0] + vsum2[4]
                + vsum3[0] + vsum3[4] + vsum4[0] + vsum4[4];
        for (; i < size; ++i)
                sum += x[i] * y[i];

        return sum;
}

float
computeSdot_sseWithPrefetching(const unsigned int size,
                               const float * __restrict__ const x,
                               const float * __restrict__ const y) {
        _mm_prefetch(x, _MM_HINT_T0);
        _mm_prefetch(y, _MM_HINT_T0);
        __m256 vsum1 = _mm256_setzero_ps();
        __m256 vsum2 = _mm256_setzero_ps();
        __m256 vsum3 = _mm256_setzero_ps();
        __m256 vsum4 = _mm256_setzero_ps();
        const auto prefetch_dist = 256;
        auto i = 0u;
        for (; i < (size & ~31); i += 32) {
                // XXX: possibly prefetching memory that isn't ours here?? w/e
                _mm_prefetch(x + prefetch_dist, _MM_HINT_T0);
                _mm_prefetch(y + prefetch_dist, _MM_HINT_T0);
                vsum1 += _mm256_load_ps(x + i) * _mm256_load_ps(y + i);
                vsum2 += _mm256_load_ps(x + i + 8) * _mm256_load_ps(y + i + 8);
                vsum1 += _mm256_load_ps(x + i + 16) * _mm256_load_ps(y + i + 16);
                vsum2 += _mm256_load_ps(x + i + 24) * _mm256_load_ps(y + i + 24);
        }

        const auto sum256_ps = [](const __m256 a) {
                return a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7];
        };

        float sum = sum256_ps(vsum1 + vsum2 + vsum3 + vsum4);
        for (; i < size; ++i)
                sum += x[i] * y[i];

        return sum;
}

#endif // MAIN1_FUNCTIONS_SDOT_H
