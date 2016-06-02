// -*- C++ -*-
// Main2_functions.h
// cs101j hw5 Problem 2
// This file contains the implementations of the integration examples

#ifndef MAIN2_FUNCTIONS_SQRT_H
#define MAIN2_FUNCTIONS_SQRT_H

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

// special include file for SIMD commands
#include <immintrin.h>

// header file for openmp
#include <omp.h>

double
integrateSqrt_scalar(const size_t numberOfIntervals,
                     const double lowerBound,
                     const double dx) {
  double integral = 0;
  for (unsigned int intervalIndex = 0;
       intervalIndex < numberOfIntervals; ++intervalIndex) {
    asm("# keep your grubby mitts out of this loop, autovectorizer!");
    const double evaluationPoint =
      lowerBound + (intervalIndex + 0.5) * dx;
    integral += std::sqrt(evaluationPoint);
  }
  return integral * dx;
}

double
integrateSqrt_compiler(const size_t numberOfIntervals,
                       const double lowerBound,
                       const double dx) {

        double integral = 0;
        for (unsigned int intervalIndex = 0;
             intervalIndex < numberOfIntervals; ++intervalIndex) {
                const double evaluationPoint =
                        lowerBound + (intervalIndex + 0.5) * dx;
                integral += std::sqrt(evaluationPoint);
        }
        return integral * dx;
}

double
integrateSqrt_manual(const size_t numberOfIntervals,
                     const double _lowerBound,
                     const double _dx) {

        __m256d integral = _mm256_set1_pd(0.);
        const __m256d lowerBounds = _mm256_set1_pd(_lowerBound);
        const __m256d dx = _mm256_set1_pd(_dx);
        const __m256d offsets = _mm256_set_pd(3., 2., 1., 0.);

        auto i = 0u;
        const auto stride = 4;
        for (; i < (numberOfIntervals & ~(stride-1)); i += stride)
                integral += _mm256_sqrt_pd(lowerBounds + (offsets + _mm256_set1_pd(i + 0.5)) * dx);

        double integral_s = integral[0] + integral[1] + integral[2] + integral[3];

        for (; i < numberOfIntervals; ++i)
                integral_s += std::sqrt(_lowerBound + (i + 0.5) * _dx);

        return integral_s * _dx;
}

#endif // MAIN2_FUNCTIONS_SQRT_H
