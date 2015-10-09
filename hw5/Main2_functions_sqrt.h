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
  // TODO: change me
  // This is simply a copy of the scalar version
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
integrateSqrt_manual(const size_t numberOfIntervals,
                     const double lowerBound,
                     const double dx) {
  // TODO: change me
  // This is simply a copy of the scalar version
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

#endif // MAIN2_FUNCTIONS_SQRT_H
