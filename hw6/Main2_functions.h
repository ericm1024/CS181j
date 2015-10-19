// -*- C++ -*-
#ifndef MAIN2_FUNCTIONS_H
#define MAIN2_FUNCTIONS_H

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

// These utilities are used on many assignments
#include "../Utilities.h"

// header file for openmp
#include <omp.h>

// header file for c++11 threads
#include <thread>

double
calculateIntegral_serial(const unsigned int numberOfThreads,
                         const unsigned int numberOfIntervals,
                         const std::array<double, 2> & integrationRange) {
  ignoreUnusedVariable(numberOfThreads);
  double integral = 0;

  const double dx =
    (integrationRange[1] - integrationRange[0]) / numberOfIntervals;
  const double base = integrationRange[0] + 0.5*dx;
  for (size_t index = 0; index < numberOfIntervals; ++index) {
    const double middleOfInterval = base + index * dx;
    integral += std::sin(middleOfInterval);
  }

  return integral;
}

double
calculateIntegral_stdThread(const unsigned int numberOfThreads,
                            const unsigned int numberOfIntervals,
                            const std::array<double, 2> & integrationRange) {
  double integral = 0;

  // TODO: calculate the integral using numberOfThreads threads

  return integral;
}

double
calculateIntegral_omp(const unsigned int numberOfThreads,
                      const unsigned int numberOfIntervals,
                      const std::array<double, 2> & integrationRange) {

  omp_set_num_threads(numberOfThreads);

  double integral = 0;

  // TODO: calculate the integral using threads

  return integral;
}

#endif // MAIN2_FUNCTIONS_H
