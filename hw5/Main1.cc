// -*- C++ -*-
// Main1.cc
// cs101j hw5 Problem 1
// An example to illustrate how to implement simple SIMD vectorization on
// vector operations.

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

using std::string;
using std::vector;
using std::array;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

// These utilities are used on many assignments
#include "../Utilities.h"

// This file contains the functions declarations for the different
//  flavors of each problem.
#include "Main1_functions_sdot.h"
#include "Main1_functions_fixedPolynomial.h"
#include "Main1_functions_offsets.h"
#include "Main1_functions_taylorExponential.h"

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

// As usual, a result checking function to make sure we have the right answer.
void
checkSdotResult(const float correctResult,
                const float testResult,
                const std::string & testName,
                const double relativeErrorTolerance) {
  char sprintfBuffer[500];
  const double absoluteError = std::abs(correctResult - testResult);
  const double relativeError = std::abs(absoluteError / correctResult);
  if (relativeError > relativeErrorTolerance) {
    sprintf(sprintfBuffer, "wrong result for sdot, "
            "it's %e but should be %e, test named "
            BOLD_ON FG_RED "%s" RESET "\n",
            testResult, correctResult, testName.c_str());
    throw std::runtime_error(sprintfBuffer);
  }
}

template <class Function>
void
__attribute__ ((noinline))
runSdotTest(const unsigned int numberOfTrials,
            const Function function,
            const unsigned int vectorSize,
            const float * const x,
            const float * const y,
            float * const sdot,
            float * const elapsedTime) {

  *elapsedTime = std::numeric_limits<float>::max();

  for (unsigned int trialNumber = 0;
       trialNumber < numberOfTrials; ++trialNumber) {
          
          // Start measuring
    const high_resolution_clock::time_point tic = high_resolution_clock::now();

    // Do the test
    *sdot = function(vectorSize, x, y);

    // Stop measuring
    const high_resolution_clock::time_point toc = high_resolution_clock::now();
    const float thisTrialsElapsedTime =
      duration_cast<duration<float> >(toc - tic).count();
    // Take the minimum values from all trials
    *elapsedTime = std::min(*elapsedTime, thisTrialsElapsedTime);
  }

}

// As usual, a result checking function to make sure we have the right answer.
void
checkArrayOfResults(const unsigned int vectorSize,
                    const float * const correctResult,
                    const float * const testResult,
                    const std::string & testName,
                    const double relativeErrorTolerance) {
  char sprintfBuffer[500];
  for (unsigned int i = 0; i < vectorSize; ++i) {
    const double absoluteError = std::abs(correctResult[i] - testResult[i]);
    const double relativeError = std::abs(absoluteError / correctResult[i]);
    if (relativeError > relativeErrorTolerance) {
      sprintf(sprintfBuffer, "wrong result for entry %u, "
              "it's %e but should be %e, test named "
              BOLD_ON FG_RED "%s" RESET "\n",
              i, testResult[i], correctResult[i], testName.c_str());
      throw std::runtime_error(sprintfBuffer);
    }
  }
}

template <class Function>
void
runFixedPolynomialTest(const unsigned int numberOfTrials,
                       const Function function,
                       const unsigned int vectorSize,
                       const float * const x,
                       const float c0,
                       const float c1,
                       const float c2,
                       const float c3,
                       float * const y,
                       float * const elapsedTime) {

  *elapsedTime = std::numeric_limits<float>::max();

  for (unsigned int trialNumber = 0;
       trialNumber < numberOfTrials; ++trialNumber) {

    // Start measuring
    const high_resolution_clock::time_point tic = high_resolution_clock::now();

    // Do the test
    function(vectorSize, x, c0, c1, c2, c3, y);

    // Stop measuring
    const high_resolution_clock::time_point toc = high_resolution_clock::now();
    const float thisTrialsElapsedTime =
      duration_cast<duration<float> >(toc - tic).count();
    // Take the minimum values from all trials
    *elapsedTime = std::min(*elapsedTime, thisTrialsElapsedTime);
  }

}

template <class Function>
void
runOffsetsTest(const unsigned int numberOfTrials,
               const Function function,
               const unsigned int vectorSize,
               const float a,
               const float b,
               const float * const x,
               const float * const y,
               const float * const z,
               float * const w,
               float * const elapsedTime) {

  *elapsedTime = std::numeric_limits<float>::max();

  for (unsigned int trialNumber = 0;
       trialNumber < numberOfTrials; ++trialNumber) {

    // Start measuring
    const high_resolution_clock::time_point tic = high_resolution_clock::now();

    // Do the test
    function(vectorSize, a, b, x, y, z, w);

    // Stop measuring
    const high_resolution_clock::time_point toc = high_resolution_clock::now();
    const float thisTrialsElapsedTime =
      duration_cast<duration<float> >(toc - tic).count();
    // Take the minimum values from all trials
    *elapsedTime = std::min(*elapsedTime, thisTrialsElapsedTime);
  }

}

template <class Function>
void
runTaylorExponentialTest(const unsigned int numberOfTrials,
                         const Function function,
                         const unsigned int vectorSize,
                         const float * const x,
                         const unsigned int numberOfTermsInExponential,
                         float * const y,
                         float * const elapsedTime) {

  *elapsedTime = std::numeric_limits<float>::max();

  for (unsigned int trialNumber = 0;
       trialNumber < numberOfTrials; ++trialNumber) {

    // Start measuring
    const high_resolution_clock::time_point tic = high_resolution_clock::now();

    // Do the test
    function(vectorSize, x, numberOfTermsInExponential, y);

    // Stop measuring
    const high_resolution_clock::time_point toc = high_resolution_clock::now();
    const float thisTrialsElapsedTime =
      duration_cast<duration<float> >(toc - tic).count();
    // Take the minimum values from all trials
    *elapsedTime = std::min(*elapsedTime, thisTrialsElapsedTime);
  }

}

int main() {

  // ===========================================================================
  // *************************** < Inputs> *************************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  const array<double, 2> vectorSizeRange = {{1e2, 1e5}};
  const unsigned int numberOfDataPoints  = 20;

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // *************************** </Inputs> *************************************
  // ===========================================================================

  // On each test, we need to make sure we get the same result.  A test will
  //  fail if the difference between any entry in our result is more than
  //  relativeErrorTolerance different than entries we got with another method.
  const double relativeErrorTolerance = 1e-3;

  // create a random number generator
  std::mt19937 randomNumberEngine;
  std::uniform_real_distribution<double> randomNumberGenerator(0, 1);

  char sprintfBuffer[500];

  const string prefix = "data/Main1_";
  const string suffix = "_shuffler";

  // Make sure that the data directory exists.
  Utilities::verifyThatDirectoryExists("data");

  // Open output files
  sprintf(sprintfBuffer, "%ssdot_results%s.csv",
          prefix.c_str(), suffix.c_str());
  FILE * sdotFile = fopen(sprintfBuffer, "w");
  sprintf(sprintfBuffer, "%sfixedPolynomial_results%s.csv",
          prefix.c_str(), suffix.c_str());
  FILE * fixedPolynomialFile = fopen(sprintfBuffer, "w");
  sprintf(sprintfBuffer, "%soffsets_results%s.csv",
          prefix.c_str(), suffix.c_str());
  FILE * offsetsFile = fopen(sprintfBuffer, "w");
  sprintf(sprintfBuffer, "%sexp_results%s.csv",
          prefix.c_str(), suffix.c_str());
  FILE * expFile = fopen(sprintfBuffer, "w");

  // For each size
  for (unsigned int dataPointIndex = 0;
       dataPointIndex < numberOfDataPoints;
       ++dataPointIndex) {

    const unsigned int vectorSize =
      Utilities::interpolateNumberLinearlyOnLogScale(vectorSizeRange[0],
                                                     vectorSizeRange[1],
                                                     numberOfDataPoints,
                                                     dataPointIndex);

    const high_resolution_clock::time_point thisSizesTic =
      high_resolution_clock::now();

    fprintf(sdotFile, "%10.4e", double(vectorSize));
    fprintf(offsetsFile, "%10.4e", double(vectorSize));
    fprintf(fixedPolynomialFile, "%10.4e", double(vectorSize));
    fprintf(expFile, "%10.4e", double(vectorSize));

    // Form the vectors
    float * x = allocateAlignedMemory<float>(vectorSize, 64);
    float * y = allocateAlignedMemory<float>(vectorSize, 64);
    float * z = allocateAlignedMemory<float>(vectorSize, 64);
    float * w = allocateAlignedMemory<float>(vectorSize, 64);
    for (size_t index = 0; index < vectorSize; ++index) {
      x[index] = randomNumberGenerator(randomNumberEngine);
      y[index] = randomNumberGenerator(randomNumberEngine);
      z[index] = randomNumberGenerator(randomNumberEngine);
      w[index] = randomNumberGenerator(randomNumberEngine);
    }

    // Calculate the number of trials for this size
    const unsigned int numberOfTrials = 
            std::max(unsigned(100), unsigned(2e6 / vectorSize));

    float elapsedTime;

    // ===============================================================
    // ********************** < do sdot > ****************************
    // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    float sdot;

    runSdotTest(numberOfTrials,
                computeSdot_scalar,
                vectorSize, x, y,
                &sdot, &elapsedTime);
    const float scalarSdot = sdot;
    fprintf(sdotFile, ", %10.4e", elapsedTime);

    runSdotTest(numberOfTrials,
                computeSdot_manual,
                vectorSize, x, y,
                &sdot, &elapsedTime);
    checkSdotResult(scalarSdot, sdot, string("sdot manual"),
                    relativeErrorTolerance);
    fprintf(sdotFile, ", %10.4e", elapsedTime);

    runSdotTest(numberOfTrials,
                computeSdot_sseWithPrefetching,
                vectorSize, x, y,
                &sdot, &elapsedTime);
    checkSdotResult(scalarSdot, sdot, string("sdot sse with prefetching"),
                    relativeErrorTolerance);
    fprintf(sdotFile, ", %10.4e", elapsedTime);

    runSdotTest(numberOfTrials,
                computeSdot_sseDotProduct,
                vectorSize, x, y,
                &sdot, &elapsedTime);
    checkSdotResult(scalarSdot, sdot, string("sdot sse dot product"),
                    relativeErrorTolerance);
    fprintf(sdotFile, ", %10.4e", elapsedTime);

    runSdotTest(numberOfTrials,
                computeSdot_compiler,
                vectorSize, x, y,
                &sdot, &elapsedTime);
    checkSdotResult(scalarSdot, sdot, string("sdot compiler"),
                    relativeErrorTolerance);
    fprintf(sdotFile, ", %10.4e", elapsedTime);

    fprintf(sdotFile, "\n");

    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // ********************** </do sdot > ****************************
    // ===============================================================


    // ===============================================================
    // ********************** < do fixed polynomial > ****************
    // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    const float c0 = 1.2;
    const float c1 = 3.4;
    const float c2 = 5.6;
    const float c3 = 7.8;

    runFixedPolynomialTest(numberOfTrials,
                           computeFixedPolynomial_scalar,
                           vectorSize, x, c0, c1, c2, c3,
                           y, &elapsedTime);
    vector<float> scalarFixedPolynomial;
    std::copy(&y[0], &y[vectorSize], std::back_inserter(scalarFixedPolynomial));
    fprintf(fixedPolynomialFile, ", %10.4e", elapsedTime);

    runFixedPolynomialTest(numberOfTrials,
                           computeFixedPolynomial_manual,
                           vectorSize, x, c0, c1, c2, c3,
                           y, &elapsedTime);
    checkArrayOfResults(vectorSize, &scalarFixedPolynomial[0], y,
                        string("fixedPolynomial manual"),
                        relativeErrorTolerance);
    fprintf(fixedPolynomialFile, ", %10.4e", elapsedTime);

    runFixedPolynomialTest(numberOfTrials,
                           computeFixedPolynomial_compiler,
                           vectorSize, x, c0, c1, c2, c3,
                           y, &elapsedTime);
    checkArrayOfResults(vectorSize, &scalarFixedPolynomial[0], y,
                        string("fixedPolynomial compiler"),
                        relativeErrorTolerance);
    fprintf(fixedPolynomialFile, ", %10.4e", elapsedTime);

    fprintf(fixedPolynomialFile, "\n");

    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // ********************** </do fixed polynomial > ****************
    // ===============================================================


    // ===============================================================
    // ********************** < do offsets > *************************
    // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    const float a = 1.2;
    const float b = 3.4;

    runOffsetsTest(numberOfTrials,
                   computeOffsets_scalar,
                   vectorSize, a, b, x, y, z,
                   w, &elapsedTime);
    vector<float> scalarOffsets;
    std::copy(&w[0], &w[vectorSize], std::back_inserter(scalarOffsets));
    fprintf(offsetsFile, ", %10.4e", elapsedTime);

    runOffsetsTest(numberOfTrials,
                   computeOffsets_scalarNoMod,
                   vectorSize, a, b, x, y, z,
                   w, &elapsedTime);
    checkArrayOfResults(vectorSize, &scalarOffsets[0], w,
                        string("offsets scalar no mod"),
                        relativeErrorTolerance);
    fprintf(offsetsFile, ", %10.4e", elapsedTime);

    runOffsetsTest(numberOfTrials,
                   computeOffsets_manual,
                   vectorSize, a, b, x, y, z,
                   w, &elapsedTime);
    checkArrayOfResults(vectorSize, &scalarOffsets[0], w,
                        string("offsets manual"),
                        relativeErrorTolerance);
    fprintf(offsetsFile, ", %10.4e", elapsedTime);

    runOffsetsTest(numberOfTrials,
                   computeOffsets_compiler,
                   vectorSize, a, b, x, y, z,
                   w, &elapsedTime);
    checkArrayOfResults(vectorSize, &scalarOffsets[0], w,
                        string("offsets compiler"),
                        relativeErrorTolerance);
    fprintf(offsetsFile, ", %10.4e", elapsedTime);

    fprintf(offsetsFile, "\n");

    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // ********************** </do offsets > *************************
    // ===============================================================

    // ===============================================================
    // ********************** < do exp > *****************************
    // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    const unsigned int numberOfTermsInExponential = 25;

    runTaylorExponentialTest(numberOfTrials,
                             computeTaylorExponential_scalar,
                             vectorSize, x, numberOfTermsInExponential,
                             y, &elapsedTime);
    vector<float> scalarTaylorExponential;
    std::copy(&y[0], &y[vectorSize], std::back_inserter(scalarTaylorExponential));
    fprintf(expFile, ", %10.4e", elapsedTime);
    std::fill(&y[0], &y[vectorSize], 0);

    runTaylorExponentialTest(numberOfTrials,
                             computeTaylorExponential_manual,
                             vectorSize, x, numberOfTermsInExponential,
                             y, &elapsedTime);
    checkArrayOfResults(vectorSize, &scalarTaylorExponential[0], y,
                        string("taylor exponential manual"),
                        relativeErrorTolerance);
    fprintf(expFile, ", %10.4e", elapsedTime);
    std::fill(&y[0], &y[vectorSize], 0);

    runTaylorExponentialTest(numberOfTrials,
                             computeTaylorExponential_compiler,
                             vectorSize, x, numberOfTermsInExponential,
                             y, &elapsedTime);
    checkArrayOfResults(vectorSize, &scalarTaylorExponential[0], y,
                        string("taylor exponential compiler"),
                        relativeErrorTolerance);
    fprintf(expFile, ", %10.4e", elapsedTime);

    fprintf(expFile, "\n");

    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // ********************** </do exp > *****************************
    // ===============================================================

    const high_resolution_clock::time_point thisSizesToc =
      high_resolution_clock::now();
    const double thisSizesElapsedTime =
      duration_cast<duration<double> >(thisSizesToc - thisSizesTic).count();
    printf("finished size %8.2e in %6.2f seconds\n", double(vectorSize),
           thisSizesElapsedTime);

    freeAlignedMemory(&x);
    freeAlignedMemory(&y);
    freeAlignedMemory(&z);
    freeAlignedMemory(&w);
  }

  fclose(sdotFile);
  fclose(offsetsFile);
  fclose(fixedPolynomialFile);
  fclose(expFile);

  return 0;
}
