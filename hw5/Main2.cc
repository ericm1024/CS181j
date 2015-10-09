// -*- C++ -*-
// Main3.cc
// cs101j hw5 Problem 2
// An example to illustrate how to implement simple SIMD vectorization on
// scalar integration

// These utilities are used on many assignments
#include "../Utilities.h"

// This file contains the functions declarations for the different
//  flavors of each problem.
#include "Main2_functions_sqrt.h"
#include "Main2_functions_fixedPolynomial.h"

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

using std::string;
using std::vector;
using std::array;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

// As usual, a result checking function to make sure we have the right answer.
void
checkResult(const double correctResult,
            const double testResult,
            const std::string & testName,
            const double relativeErrorTolerance) {
  char sprintfBuffer[500];
  const double absoluteError = std::abs(correctResult - testResult);
  const double relativeError = std::abs(absoluteError / correctResult);
  if (relativeError > relativeErrorTolerance) {
    sprintf(sprintfBuffer, "wrong result, "
            "it's %e but should be %e, test named "
            BOLD_ON FG_RED "%s" RESET "\n",
            testResult, correctResult, testName.c_str());
    throw std::runtime_error(sprintfBuffer);
  }
}

template <class Function>
void
runSqrtTest(const unsigned int numberOfTrials,
            const Function function,
            const unsigned int numberOfIntervals,
            const double lowerBound,
            const double dx,
            double * const result,
            double * const elapsedTime) {

  *elapsedTime = std::numeric_limits<double>::max();

  for (unsigned int trialNumber = 0;
       trialNumber < numberOfTrials; ++trialNumber) {

    // Start measuring
    const high_resolution_clock::time_point tic = high_resolution_clock::now();

    // Do the test
    *result = function(numberOfIntervals, lowerBound, dx);

    // Stop measuring
    const high_resolution_clock::time_point toc = high_resolution_clock::now();
    const double thisTrialsElapsedTime =
      duration_cast<duration<double> >(toc - tic).count();
    // Take the minimum values from all trials
    *elapsedTime = std::min(*elapsedTime, thisTrialsElapsedTime);
  }

}

template <class Function>
void
runFixedPolynomialTest(const unsigned int numberOfTrials,
                       const Function function,
                       const unsigned int numberOfIntervals,
                       const double lowerBound,
                       const double dx,
                       const double c0,
                       const double c1,
                       const double c2,
                       const double c3,
                       double * const result,
                       double * const elapsedTime) {

  *elapsedTime = std::numeric_limits<double>::max();

  for (unsigned int trialNumber = 0;
       trialNumber < numberOfTrials; ++trialNumber) {

    // Start measuring
    const high_resolution_clock::time_point tic = high_resolution_clock::now();

    // Do the test
    *result = function(numberOfIntervals, lowerBound, dx, c0, c1, c2, c3);

    // Stop measuring
    const high_resolution_clock::time_point toc = high_resolution_clock::now();
    const double thisTrialsElapsedTime =
      duration_cast<duration<double> >(toc - tic).count();
    // Take the minimum values from all trials
    *elapsedTime = std::min(*elapsedTime, thisTrialsElapsedTime);
  }

}

int main() {

  // ===========================================================================
  // *************************** < Inputs> *************************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  const array<double, 2> numberOfIntervalsRange = {{1e3, 1e7}};
  const unsigned int numberOfDataPoints  = 20;
  // The integration bounds is not interesting, but making the
  //  beginning nonzero helps us make sure that code is behaving
  //  correctly.
  const array<double, 2> integrationBounds = {{.61, 1.314}};

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // *************************** </Inputs> *************************************
  // ===========================================================================

  // On each test, we need to make sure we get the same result.  A test will
  //  fail if the difference between any entry in our result is more than
  //  relativeErrorTolerance different than entries we got with another method.
  const double relativeErrorTolerance = 1e-3;

  const string prefix = "data/Main2_";
  const string suffix = "_shuffler";

  char sprintfBuffer[500];
  sprintf(sprintfBuffer, "%ssqrt_results%s.csv", prefix.c_str(), suffix.c_str());
  FILE * sqrtFile = fopen(sprintfBuffer, "w");
  sprintf(sprintfBuffer, "%sfixedPolynomial_results%s.csv",
          prefix.c_str(), suffix.c_str());
  FILE * fixedPolynomialFile = fopen(sprintfBuffer, "w");

  // For each numberOfIntervals
  for (unsigned int dataPointIndex = 0;
       dataPointIndex < numberOfDataPoints;
       ++dataPointIndex) {

    const unsigned int numberOfIntervals =
      Utilities::interpolateNumberLinearlyOnLogScale(numberOfIntervalsRange[0],
                                                     numberOfIntervalsRange[1],
                                                     numberOfDataPoints,
                                                     dataPointIndex);

    const high_resolution_clock::time_point thisSizesTic =
      high_resolution_clock::now();

    const double dx =
      (integrationBounds[1] - integrationBounds[0]) / numberOfIntervals;

    const unsigned int numberOfTrials =
      std::max(unsigned(2), unsigned(1e7 / numberOfIntervals));

    fprintf(sqrtFile, "%10.4e", double(numberOfIntervals));
    fprintf(fixedPolynomialFile, "%10.4e", double(numberOfIntervals));

    double elapsedTime;

    // ===============================================================
    // ********************** < do sqrt > *****************************
    // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    double sqrtResult;

    runSqrtTest(numberOfTrials,
                integrateSqrt_scalar,
                numberOfIntervals, integrationBounds[0], dx,
                &sqrtResult, &elapsedTime);
    const double scalarSqrtResult = sqrtResult;
    fprintf(sqrtFile, ", %10.4e", elapsedTime);

    runSqrtTest(numberOfTrials,
                integrateSqrt_manual,
                numberOfIntervals, integrationBounds[0], dx,
                &sqrtResult, &elapsedTime);
    checkResult(scalarSqrtResult, sqrtResult, string("sqrt manual"),
                relativeErrorTolerance);
    fprintf(sqrtFile, ", %10.4e", elapsedTime);

    runSqrtTest(numberOfTrials,
                integrateSqrt_compiler,
                numberOfIntervals, integrationBounds[0], dx,
                &sqrtResult, &elapsedTime);
    checkResult(scalarSqrtResult, sqrtResult, string("sqrt compiler"),
                relativeErrorTolerance);
    fprintf(sqrtFile, ", %10.4e", elapsedTime);

    fprintf(sqrtFile, "\n");

    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // ********************** </do sqrt > ****************************
    // ===============================================================

    // ===============================================================
    // ********************** < do fixed polynomial > ****************
    // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    const double c0 = 1;
    const double c1 = 1;
    const double c2 = 1;
    const double c3 = 1;

    double fixedPolynomialResult;

    runFixedPolynomialTest(numberOfTrials,
                           integrateFixedPolynomial_scalar,
                           numberOfIntervals, integrationBounds[0], dx,
                           c0, c1, c2, c3,
                           &fixedPolynomialResult, &elapsedTime);
    const double scalarFixedPolynomialResult = fixedPolynomialResult;
    fprintf(fixedPolynomialFile, ", %10.4e", elapsedTime);

    runFixedPolynomialTest(numberOfTrials,
                           integrateFixedPolynomial_manual,
                           numberOfIntervals, integrationBounds[0], dx,
                           c0, c1, c2, c3,
                           &fixedPolynomialResult, &elapsedTime);
    checkResult(scalarFixedPolynomialResult, fixedPolynomialResult,
                string("fixedPolynomial manual"), relativeErrorTolerance);
    fprintf(fixedPolynomialFile, ", %10.4e", elapsedTime);

    runFixedPolynomialTest(numberOfTrials,
                           integrateFixedPolynomial_compiler,
                           numberOfIntervals, integrationBounds[0], dx,
                           c0, c1, c2, c3,
                           &fixedPolynomialResult, &elapsedTime);
    checkResult(scalarFixedPolynomialResult, fixedPolynomialResult,
                string("fixedPolynomial compiler"), relativeErrorTolerance);
    fprintf(fixedPolynomialFile, ", %10.4e", elapsedTime);

    fprintf(fixedPolynomialFile, "\n");

    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // ********************** </do fixed polynomial > ****************
    // ===============================================================

    const high_resolution_clock::time_point thisSizesToc =
      high_resolution_clock::now();
    const double thisSizesElapsedTime =
      duration_cast<duration<double> >(thisSizesToc - thisSizesTic).count();
    printf("finished size %8.2e in %6.2f seconds\n", double(numberOfIntervals),
           thisSizesElapsedTime);

  }

  fclose(sqrtFile);
  fclose(fixedPolynomialFile);

  return 0;
}
