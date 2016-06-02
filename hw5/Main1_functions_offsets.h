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
        auto i = 0u;
        for (; i < 8; ++i)
                w[i] = a * x[i] * y[i + 16] / (b * z[i + size - 8]);
        
        for (; i < size - 16; ++i)
                w[i] = a * x[i] * y[i + 16] / (b * z[i - 8]);

        for (; i < size; ++i)
                w[i] = a * x[i] * y[i + 16 - size] / (b * z[i - 8]);
}

void
computeOffsets_compiler(const unsigned int size,
                        const float a,
                        const float b,
                        const float * __restrict__ const _x,
                        const float * __restrict__ const _y,
                        const float * __restrict__ const _z,
                        float * __restrict__ const _w) {

        const float *x = (const float *)__builtin_assume_aligned(_x, 16);
        const float *y = (const float *)__builtin_assume_aligned(_y, 16);
        const float *z = (const float *)__builtin_assume_aligned(_z, 16);
        float *w = (float *)__builtin_assume_aligned(_w, 16);

        auto i = 0u;
        for (; i < 8; ++i)
                w[i] = a * x[i] * y[i + 16] / (b * z[i + size - 8]);

        y += 16;
        z -= 8;
        for (auto j = i; j < size - 16; ++j)
                w[j] = a * x[j] * y[j] / (b * z[j]);

        y -= 16;
        z += 8;        
        for (i = size-16; i < size; ++i)
                w[i] = a * x[i] * y[i + 16 - size] / (b * z[i - 8]);
}

void
computeOffsets_manual(const unsigned int size,
                      const float _a,
                      const float _b,
                      const float * __restrict__ const x,
                      const float * __restrict__ const y,
                      const float * __restrict__ const z,
                      float * __restrict__ const w) {

        __m256 a = _mm256_set1_ps(_a);
        __m256 b = _mm256_set1_ps(_b);

        auto i = 0u;
        _mm256_store_ps(w + i, a * _mm256_load_ps(x) * _mm256_load_ps(y + 16)
                        / (b * _mm256_loadu_ps(z + size - 8)));
        i += 8;

        const auto stride = 16;
        const auto bound = std::min(size - (16 + stride),  size & ~(stride-1));
        for (; i < bound; i += 8) {
                _mm256_store_ps(w + i, a * _mm256_load_ps(x+i) * _mm256_load_ps(y + i + 16)
                                * _mm256_rcp_ps(b * _mm256_load_ps(z + i - 8)));

                i += 8;
                _mm256_store_ps(w + i, a * _mm256_load_ps(x+i) * _mm256_load_ps(y + i + 16)
                                * _mm256_rcp_ps(b * _mm256_load_ps(z + i - 8)));
        }

        for (; i < size - 16; ++i)
                w[i] = _a * x[i] * y[i + 16] / (_b * z[i - 8]);
        
        for (; i < size; ++i)
                w[i] = _a * x[i] * y[i + 16 - size] / (_b * z[i - 8]);
}

#endif // MAIN1_FUNCTIONS_OFFSETS_H
