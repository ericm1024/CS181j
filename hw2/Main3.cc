// -*- C++ -*-
// Main3.cc
// cs181j hw2 Problem 3
// This is an exercise in measuring the effect of memoization

// c junk
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdio>

// c++ junk
#include <array>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <random>
#include <fstream>

// These utilities are used on many assignments
#include "../Utilities.h"

// Students have given feedback that they would like the functors
//  they're supposed to work on to be split into another file, so here
//  it is.
#include "Main3_functors.h"

// Only bring in from the standard namespace things that we care about.
// Remember, it's a naughty thing to just use the whole namespace.
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::vector;
using std::string;
using std::array;

// This utility returns the maximum difference between the entries in
//  a correct result and a test result.
void
checkResult(const vector<double> & correctResult,
            const vector<double> & result,
            const double absoluteErrorTolerance,
            const string & methodName) {
  char sprintfBuffer[500];
  if (correctResult.size() != result.size()) {
    sprintf(sprintfBuffer, "invalid sizes for computing the difference between "
            "results, method name %s\n", methodName.c_str());
    throw std::runtime_error(sprintfBuffer);
  }
  double maxError = 0;
  for (unsigned int index = 0; index < result.size(); ++index) {
    maxError =
      std::max(maxError,
               std::abs(correctResult[index] - result[index]));
  }
  if (maxError > absoluteErrorTolerance) {
    sprintf(sprintfBuffer, "different results detected: "
            "the largest difference in an entry is %11.4e, method named %s\n",
            maxError, methodName.c_str());
    throw std::runtime_error(sprintfBuffer);
  }
}

template <class Calculator>
void
runTimingTest(const unsigned int numberOfTrials,
              const vector<double> & input,
              Calculator * calculator,
              vector<double> * result,
              double * elapsedTime) {

  *elapsedTime = std::numeric_limits<double>::max();

  for (unsigned int trialNumber = 0;
       trialNumber < numberOfTrials; ++trialNumber) {

    // Clear the cache
    Utilities::clearCpuCache();

    // Start timing
    const high_resolution_clock::time_point tic =
      high_resolution_clock::now();

    // Run the test
    calculator->computePowers(input, result);

    // Stop timing
    const high_resolution_clock::time_point toc =
      high_resolution_clock::now();
    *elapsedTime =
      std::min(*elapsedTime,
               duration_cast<duration<double> >(toc - tic).count());
  }

}
int main() {

  // ===========================================================================
  // *************************** < Inputs> *************************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  // In this example, we calculate various integral powers of a bunch
  //  of inputs, like y[i] = x[i]^13
  // The power controls the amount of work we do per input.
  // We will calculate inputSize inputs (values of i,
  //  so the length of the input and output arrays).
  const unsigned int inputSize = 1e4;
  // This is a duplication rate.  When it's 0, then every value of x[i]
  //  will be different.  When it's 0.5, then 50% of the values are repeated.
  //  When it's 1.0, then every input value is the same.
  //const vector<double> duplicationRates = {{0.000, 0.250, 0.500, 0.750, 0.850,
  //0.900, 0.925, 0.950, 0.975, 1.000}};
  const vector<double> duplicationRates = {{0.000, 0.330, 0.660, 0.900, 0.950,
                                            0.980, 0.990, 0.995, 1.000}};
  // This is a the range of the power to compute.  When it's 2, we just
  //  calculate x[i]^2 for every input.  When it's 17, we calculate x[i]^17
  //const vector<double> powers = {{1, 2, 3, 4, 5, 6, 8, 10, 14, 18, 20}};
  const vector<double> powers = {{1, 2, 3, 5, 10, 15, 20}};
  // This is the resolution used to determine if we can used a stored result.
  // That is, if a current query is within this resolution of a stored query,
  //  we use the stored query.
  const double memoizationResolution = 1e-7;
  // We do each version this many times and take the minimum runtime, to
  //  smooth curves out.
  const unsigned int numberOfTrials = 10;

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // *************************** </Inputs> *************************************
  // ===========================================================================
  const unsigned int numberOfDuplicationRateDataPoints = duplicationRates.size();
  const unsigned int numberOfPowerDataPoints = powers.size();

  // On each test, we need to make sure we get the same result.  A test will
  //  fail if the difference between any entry in our result is more than
  //  absoluteErrorTolerance different than entries we got with another method.
  const double absoluteErrorTolerance = 1e-4;

  // Make sure that the data directory exists.
  Utilities::verifyThatDirectoryExists("data");

  // This prefix and suffix will determine where files will be written and
  //  their names.
  const string prefix = "data/Main3_";
  const string suffix = "_shuffler";

  vector<vector<double> >
    duplicationRateOutputMatrix(numberOfDuplicationRateDataPoints,
                                vector<double>(numberOfPowerDataPoints, 0));
  vector<vector<double> >
    powerOutputMatrix(numberOfDuplicationRateDataPoints,
                      vector<double>(numberOfPowerDataPoints, 0));
  vector<vector<double> >
    unMemoizedOutputMatrix(numberOfDuplicationRateDataPoints,
                           vector<double>(numberOfPowerDataPoints, 0));
  vector<vector<double> >
    mapMemoizedOutputMatrix(numberOfDuplicationRateDataPoints,
                            vector<double>(numberOfPowerDataPoints, 0));
  vector<vector<double> >
    arrayMemoizedOutputMatrix(numberOfDuplicationRateDataPoints,
                              vector<double>(numberOfPowerDataPoints, 0));

  vector<double> input(inputSize);
  // Make a random number generator
  std::default_random_engine randomNumberGenerator;
  std::uniform_real_distribution<float> uniformRealDistribution(0, 0.99);

  // For each duplication rate
  for (unsigned int duplicationRateIndex = 0;
       duplicationRateIndex < numberOfDuplicationRateDataPoints;
       ++duplicationRateIndex) {

    const double duplicationRate = duplicationRates[duplicationRateIndex];

    // Start timing, simply for a heartbeat message
    const high_resolution_clock::time_point thisSizesTic =
      high_resolution_clock::now();

    // Prepare the inputs
    const unsigned int numberOfUniqueInputs =
      std::max((unsigned)1, (unsigned)(inputSize * (1. - duplicationRate)));
    std::uniform_int_distribution<unsigned>
      uniformIntDistribution(0, numberOfUniqueInputs - 1);
    // Generate the unique values
    unsigned int inputIndex = 0;
    for (; inputIndex < numberOfUniqueInputs; ++inputIndex) {
      input[inputIndex] = uniformRealDistribution(randomNumberGenerator);
    }
    // Now do the duplication
    for (; inputIndex < inputSize; ++inputIndex) {
      input[inputIndex] = input[uniformIntDistribution(randomNumberGenerator)];
    }
    // input is now ready to use for this duplication rate.

    // For each power
    for (unsigned int powerIndex = 0;
         powerIndex < numberOfPowerDataPoints;
         ++powerIndex) {

      const unsigned int power = powers[powerIndex];

      vector<double> unMemoizedResult(inputSize,
                                      std::numeric_limits<double>::quiet_NaN());
      double unMemoizedElapsedTime;
      PowerCalculator unMemoizedCalculator(power);
      runTimingTest(numberOfTrials,
                    input,
                    &unMemoizedCalculator,
                    &unMemoizedResult,
                    &unMemoizedElapsedTime);
      unMemoizedOutputMatrix[duplicationRateIndex][powerIndex] =
        unMemoizedElapsedTime;

      vector<double> mapMemoizedResult(inputSize,
                                       std::numeric_limits<double>::quiet_NaN());
      double mapMemoizedElapsedTime;
      MapMemoizedPowerCalculator mapMemoizedCalculator(power,
                                                       memoizationResolution);
      runTimingTest(numberOfTrials,
                    input,
                    &mapMemoizedCalculator,
                    &mapMemoizedResult,
                    &mapMemoizedElapsedTime);
      // Check the result
      checkResult(unMemoizedResult,
                  mapMemoizedResult,
                  absoluteErrorTolerance,
                  string("map memoized"));
      mapMemoizedOutputMatrix[duplicationRateIndex][powerIndex] =
        mapMemoizedElapsedTime;

      vector<double> arrayMemoizedResult(inputSize,
                                         std::numeric_limits<double>::quiet_NaN());
      double arrayMemoizedElapsedTime;
      ArrayMemoizedPowerCalculator arrayMemoizedCalculator(power,
                                                           memoizationResolution);
      runTimingTest(numberOfTrials,
                    input,
                    &arrayMemoizedCalculator,
                    &arrayMemoizedResult,
                    &arrayMemoizedElapsedTime);
      // Check the result
      checkResult(unMemoizedResult,
                  arrayMemoizedResult,
                  absoluteErrorTolerance,
                  string("array memoized"));
      arrayMemoizedOutputMatrix[duplicationRateIndex][powerIndex] =
        arrayMemoizedElapsedTime;

      duplicationRateOutputMatrix[duplicationRateIndex][powerIndex] =
        duplicationRate;
      powerOutputMatrix[duplicationRateIndex][powerIndex] =
        power;
    }

    // Output a heartbeat message
    const high_resolution_clock::time_point toc = high_resolution_clock::now();
    const double thisSizesElapsedTime =
      duration_cast<duration<double> >(toc - thisSizesTic).count();
    printf("finished duplication rate %4.2f (%2u/%2u) with %3u repeats "
           "in %8.2e seconds\n",
           duplicationRate, duplicationRateIndex,
           numberOfDuplicationRateDataPoints,
           numberOfTrials, thisSizesElapsedTime);
  }

  // Write files
  Utilities::writeMatrixToFile(duplicationRateOutputMatrix,
                               prefix + string("duplicationRate") + suffix);
  Utilities::writeMatrixToFile(powerOutputMatrix,
                               prefix + string("power") + suffix);
  Utilities::writeMatrixToFile(unMemoizedOutputMatrix,
                               prefix + string("unMemoized") + suffix);
  Utilities::writeMatrixToFile(mapMemoizedOutputMatrix,
                               prefix + string("mapMemoized") + suffix);
  Utilities::writeMatrixToFile(arrayMemoizedOutputMatrix,
                               prefix + string("arrayMemoized") + suffix);

  return 0;
}
