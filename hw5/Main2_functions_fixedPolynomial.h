// -*- C++ -*-
// Main2_functions.h
// cs101j hw5 Problem 2
// This file contains the implementations of the integration examples

#ifndef MAIN2_FUNCTIONS_FIXEDPOLYNOMIAL_H
#define MAIN2_FUNCTIONS_FIXEDPOLYNOMIAL_H

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
integrateFixedPolynomial_scalar(const size_t numberOfIntervals,
                                const double lowerBound,
                                const double dx,
                                const double c0,
                                const double c1,
                                const double c2,
                                const double c3) {
  double integral = 0;
  for (size_t intervalIndex = 0;
       intervalIndex < numberOfIntervals; ++intervalIndex) {
    asm("# keep your grubby mitts out of this loop, autovectorizer!");
    const double value = lowerBound + (intervalIndex + 0.5) * dx;
    integral += c0 + c1 * value + c2 * value * value +
      c3 * value * value * value;
  }
  return integral * dx;
}

double
integrateFixedPolynomial_compiler(const size_t numberOfIntervals,
                                  const double lowerBound,
                                  const double dx,
                                  const double c0,
                                  const double c1,
                                  const double c2,
                                  const double c3) {
  // TODO: change me
  // This is simply a copy of the scalar version
  double integral = 0;
  for (size_t intervalIndex = 0;
       intervalIndex < numberOfIntervals; ++intervalIndex) {
    const double value = lowerBound + (intervalIndex + 0.5) * dx;
    integral += c0 + c1 * value + c2 * value * value +
      c3 * value * value * value;
  }
  return integral * dx;
}

double
integrateFixedPolynomial_manual(const size_t numberOfIntervals,
                                const double lowerBound,
                                const double dx,
                                const double c0,
                                const double c1,
                                const double c2,
                                const double c3) {
  // TODO: change me
  // This is simply a copy of the scalar version
  double integral = 0;
  for (size_t intervalIndex = 0;
       intervalIndex < numberOfIntervals; ++intervalIndex) {
    const double value = lowerBound + (intervalIndex + 0.5) * dx;
    integral += c0 + c1 * value + c2 * value * value +
      c3 * value * value * value;
  }
  return integral * dx;
}

#endif // MAIN2_FUNCTIONS_FIXEDPOLYNOMIAL_H
