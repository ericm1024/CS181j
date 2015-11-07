// -*- C++ -*-
// Main2_functions.h
// cs181j hw8 Problem 2
// The functors used in the Main2.cc driver.

// These utilities are used on many assignments
#include "../Utilities.h"

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

// header file for std::thread
#include <thread>

// header file for openmp
#include <omp.h>

// header files for tbb
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_scheduler_init.h>

double
calculateOnePolynomial(const double thisInput,
                       const std::vector<double> & coefficients,
                       const unsigned int thisInputsPolynomialOrder) {
  // assume polynomial orders are multiples 4 just so that we can do
  //  bucketing easily.
  double currentPower0 = 1;
  double currentPower1 = thisInput;
  double currentPower2 = thisInput * thisInput;
  double currentPower3 = thisInput * thisInput * thisInput;
  const double currentPowerMultiplier =
    thisInput * thisInput * thisInput * thisInput;
  double sum0 = 0;
  double sum1 = 0;
  double sum2 = 0;
  double sum3 = 0;
  for (unsigned int powerIndex = 0;
       powerIndex < thisInputsPolynomialOrder; powerIndex += 4) {
    sum0 += coefficients[powerIndex + 0] * currentPower0;
    currentPower0 *= currentPowerMultiplier;
    sum1 += coefficients[powerIndex + 1] * currentPower1;
    currentPower1 *= currentPowerMultiplier;
    sum2 += coefficients[powerIndex + 2] * currentPower2;
    currentPower2 *= currentPowerMultiplier;
    sum3 += coefficients[powerIndex + 3] * currentPower3;
    currentPower3 *= currentPowerMultiplier;
  }
  return sum0 + sum1 + sum2 + sum3;
}

double
calculateSumOfPolynomials_serial(const unsigned int numberOfThreads,
                                 const std::vector<double> & input,
                                 const std::vector<double> & coefficients,
                                 const PolynomialOrderStyle polynomialOrderStyle) {
  ignoreUnusedVariable(numberOfThreads);
  double sum = 0;

  const double doubleInputSize = double(input.size());
  const unsigned int inputSize = input.size();
  const unsigned int maxPolynomialOrder = coefficients.size();
  const unsigned int maxPolynomialOrderOver4 = coefficients.size() / 4;
  for (unsigned int inputIndex = 0;
       inputIndex < inputSize; ++inputIndex) {
    const unsigned int thisInputsPolynomialOrder =
      (polynomialOrderStyle == PolynomialOrderFixed) ?
      maxPolynomialOrder :
      ((inputIndex/doubleInputSize) * maxPolynomialOrderOver4) * 4;
    sum += calculateOnePolynomial(input[inputIndex],
                                  coefficients,
                                  thisInputsPolynomialOrder);
  }

  return sum;
}

double
calculateSumOfPolynomials_omp(const unsigned int numberOfThreads,
                              const std::vector<double> & input,
                              const std::vector<double> & coefficients,
                              const PolynomialOrderStyle polynomialOrderStyle) {
  ignoreUnusedVariable(numberOfThreads);
  double sum = 0;

  // TODO: replace this with an openmp version
  sum = calculateSumOfPolynomials_serial(numberOfThreads,
                                         input,
                                         coefficients,
                                         polynomialOrderStyle);

  return sum;
}

double
calculateSumOfPolynomials_tbb(const unsigned int numberOfThreads,
                              const std::vector<double> & input,
                              const std::vector<double> & coefficients,
                              const PolynomialOrderStyle polynomialOrderStyle) {
  // Ignore this number of threads, I set it in the main function
  ignoreUnusedVariable(numberOfThreads);

  double sum = 0;

  // TODO: replace this with a tbb version
  sum = calculateSumOfPolynomials_serial(numberOfThreads,
                                         input,
                                         coefficients,
                                         polynomialOrderStyle);

  return sum;
}
