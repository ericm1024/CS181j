// -*- C++ -*-
// Main2.cc
// cs181j hw8 Problem 2
// An example to illustrate how to implement simple threading for a polynomial
//  calculation with tbb.  We then explore load balancing by changing the
//  polynomial order for each evaluation.

// These utilities are used on many assignments
#include "../Utilities.h"

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

#include "Main2_functions.h"

using std::string;
using std::vector;
using std::array;
using std::size_t;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

void
checkResult(const double correctResult,
            const double testResult,
            const string & testName,
            const double relativeErrorTolerance) {
  char sprintfBuffer[500];
  const double absoluteError = std::abs(testResult - correctResult);
  const double relativeError = absoluteError / correctResult;
  if (relativeError > relativeErrorTolerance) {
    sprintf(sprintfBuffer, "threads style %s, incorrect testResult of %e, "
            "correct is %e, off by %e and tolerance is %e\n",
            testName.c_str(),
            testResult, correctResult,
            relativeError, relativeErrorTolerance);
    throw std::runtime_error(sprintfBuffer);
  }
}

template <class Function>
void
runTimingTest(const unsigned int numberOfTrials,
              const Function function,
              const unsigned int numberOfThreads,
              const vector<double> & input,
              const vector<double> & coefficients,
              const PolynomialOrderStyle polynomialOrderStyle,
              double * result,
              double * elapsedTime) {

  *elapsedTime = std::numeric_limits<double>::max();

  for (unsigned int trialNumber = 0;
       trialNumber < numberOfTrials; ++trialNumber) {

    // Start measuring
    const high_resolution_clock::time_point tic = high_resolution_clock::now();

    // Do the calculation
    *result = function(numberOfThreads, input, coefficients, polynomialOrderStyle);

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

  // A lot of homeworks will run something over a range of sizes,
  //  which will then be plotted by some script.
  // This controls the input size
  const array<double, 2> rangeOfNumberOfInputs = {{1e2, 1e5}};
  // This number controls how many data points are made and plotted.
  const unsigned int numberOfDataPoints = 13;
  const vector<unsigned int> numbersOfThreads =
    {{1, 2, 4, 8, 12, 18, 24, 36, 48}};
  const unsigned int maxPolynomialOrder = 100;
  // This controls what type of polynomial order is used: is it fixed
  //  (independent of the index, so every polynomial costs the same) or
  //  proportional to the index, where every polynomial costs a different
  //  amount.
  const array<PolynomialOrderStyle, 2> polynomialOrderStyles =
    {{PolynomialOrderFixed, PolynomialOrderProportionalToIndex}};

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // *************************** </Inputs> *************************************
  // ===========================================================================

  const double relativeErrorTolerance = 1e-5;

  std::mt19937 randomNumberEngine;
  std::uniform_real_distribution<double> randomNumberGenerator(0, 1);

  // for each polynomial order style
  for (const PolynomialOrderStyle polynomialOrderStyle :
         polynomialOrderStyles) {

    printf("processing a polynomial order style of %s\n",
           convertPolynomialOrderStyleToString(polynomialOrderStyle).c_str());

    char sprintfBuffer[500];
    sprintf(sprintfBuffer, "data/Main2_%s_",
            convertPolynomialOrderStyleToString(polynomialOrderStyle).c_str());
    const string prefix = sprintfBuffer;
    const string suffix = "_shuffler";

    // create output matrices for plotting
    vector<vector<double> >
      inputSizeMatrixForPlotting(numberOfDataPoints,
                                 vector<double>(numbersOfThreads.size(), 0));
    vector<vector<double> >
      numberOfThreadsMatrixForPlotting(numberOfDataPoints,
                                       vector<double>(numbersOfThreads.size(), 0));
    vector<vector<double> >
      serialTimes(numberOfDataPoints,
                  vector<double>(numbersOfThreads.size(), 0));
#ifdef ENABLE_STD_THREAD
    vector<vector<double> >
      stdThreadTimes(numberOfDataPoints,
                     vector<double>(numbersOfThreads.size(), 0));
#endif
    vector<vector<double> >
      ompStaticTimes(numberOfDataPoints,
                     vector<double>(numbersOfThreads.size(), 0));
    vector<vector<double> >
      ompDynamicTimes(numberOfDataPoints,
                      vector<double>(numbersOfThreads.size(), 0));
    vector<vector<double> >
      ompGuidedTimes(numberOfDataPoints,
                     vector<double>(numbersOfThreads.size(), 0));
    vector<vector<double> >
      tbbTimes(numberOfDataPoints,
               vector<double>(numbersOfThreads.size(), 0));

    // for each size
    for (unsigned int dataPointIndex = 0;
         dataPointIndex < numberOfDataPoints;
         ++dataPointIndex) {

      const unsigned int inputSize =
        Utilities::interpolateNumberLinearlyOnLogScale(rangeOfNumberOfInputs[0],
                                                       rangeOfNumberOfInputs[1],
                                                       numberOfDataPoints,
                                                       dataPointIndex);

      const high_resolution_clock::time_point thisSizesTic =
        high_resolution_clock::now();

      // calculate the number of repeats
      const unsigned int numberOfTrials =
        std::min(unsigned(500),
                 std::max(unsigned(10), unsigned(5e5 / inputSize)));

      // prepare inputs and coefficients
      vector<double> coefficients(maxPolynomialOrder);
      for (size_t index = 0; index < maxPolynomialOrder; ++index) {
        coefficients[index] = randomNumberGenerator(randomNumberEngine);
      }
      vector<double> input(inputSize);
      for (size_t index = 0; index < inputSize; ++index) {
        input[index] = randomNumberGenerator(randomNumberEngine);
      }

      // run the serial version to get baseline numbers
      double serialResult;
      double serialElapsedTime;
      const unsigned int numberOfThreadsForSerial = 1;
      runTimingTest(numberOfTrials,
                    calculateSumOfPolynomials_serial,
                    numberOfThreadsForSerial, // this is ignored
                    input,
                    coefficients,
                    polynomialOrderStyle,
                    &serialResult,
                    &serialElapsedTime);

      for (unsigned int numberOfThreadsIndex = 0;
           numberOfThreadsIndex < numbersOfThreads.size();
           ++numberOfThreadsIndex) {
        const unsigned int numberOfThreads =
          numbersOfThreads[numberOfThreadsIndex];

        try {

          // set the number of threads for omp and tbb
          omp_set_num_threads(numberOfThreads);
          tbb::task_scheduler_init init(numberOfThreads);

          serialTimes[dataPointIndex][numberOfThreadsIndex] = serialElapsedTime;

          // std thread version
          double result;
#ifdef ENABLE_STD_THREAD
          runTimingTest(numberOfTrials,
                        calculateSumOfPolynomials_stdThread,
                        numberOfThreads,
                        input,
                        coefficients,
                        polynomialOrderStyle,
                        &result,
                        &stdThreadTimes[dataPointIndex][numberOfThreadsIndex]);
          checkResult(serialResult,
                      result,
                      string("std thread"),
                      relativeErrorTolerance);
#endif

          const int aValueOfZeroMeansUseTheDefaultChunkSizeForThisSchedule = 0;

          // openmp static version
          omp_set_schedule(omp_sched_static,
                           aValueOfZeroMeansUseTheDefaultChunkSizeForThisSchedule);
          runTimingTest(numberOfTrials,
                        calculateSumOfPolynomials_omp,
                        numberOfThreads,
                        input,
                        coefficients,
                        polynomialOrderStyle,
                        &result,
                        &ompStaticTimes[dataPointIndex][numberOfThreadsIndex]);
          checkResult(serialResult,
                      result,
                      string("omp static"),
                      relativeErrorTolerance);

          // openmp dynamic version
          omp_set_schedule(omp_sched_dynamic,
                           aValueOfZeroMeansUseTheDefaultChunkSizeForThisSchedule);
          runTimingTest(numberOfTrials,
                        calculateSumOfPolynomials_omp,
                        numberOfThreads,
                        input,
                        coefficients,
                        polynomialOrderStyle,
                        &result,
                        &ompDynamicTimes[dataPointIndex][numberOfThreadsIndex]);
          checkResult(serialResult,
                      result,
                      string("omp dynamic"),
                      relativeErrorTolerance);

          // openmp guided version
          omp_set_schedule(omp_sched_guided,
                           aValueOfZeroMeansUseTheDefaultChunkSizeForThisSchedule);
          runTimingTest(numberOfTrials,
                        calculateSumOfPolynomials_omp,
                        numberOfThreads,
                        input,
                        coefficients,
                        polynomialOrderStyle,
                        &result,
                        &ompGuidedTimes[dataPointIndex][numberOfThreadsIndex]);
          checkResult(serialResult,
                      result,
                      string("omp guided"),
                      relativeErrorTolerance);

          // tbb auto version
          runTimingTest(numberOfTrials,
                        calculateSumOfPolynomials_tbb,
                        numberOfThreads,
                        input,
                        coefficients,
                        polynomialOrderStyle,
                        &result,
                        &tbbTimes[dataPointIndex][numberOfThreadsIndex]);
          checkResult(serialResult,
                      result,
                      string("tbb"),
                      relativeErrorTolerance);
        } catch (const std::exception & e) {
          fprintf(stderr, "exception caught attempting size %u (index %u) "
                  "with %u threads (index %u)\n",
                  inputSize, dataPointIndex, numberOfThreads,
                  numberOfThreadsIndex);
          throw;
        }

        inputSizeMatrixForPlotting[dataPointIndex][numberOfThreadsIndex] =
          inputSize;
        numberOfThreadsMatrixForPlotting[dataPointIndex][numberOfThreadsIndex] =
          numberOfThreads;
      }

      const high_resolution_clock::time_point thisSizesToc =
        high_resolution_clock::now();
      const double thisSizesElapsedTime =
        duration_cast<duration<double> >(thisSizesToc - thisSizesTic).count();
      printf("finished size %8.2e with %8.2e trials in %6.2f seconds\n",
             double(inputSize), double(numberOfTrials),
             thisSizesElapsedTime);
    }


    Utilities::writeMatrixToFile(inputSizeMatrixForPlotting, prefix + string("inputSize") + suffix);
    Utilities::writeMatrixToFile(numberOfThreadsMatrixForPlotting, prefix + string("numberOfThreads") + suffix);
    Utilities::writeMatrixToFile(serialTimes, prefix + string("serialTimes") + suffix);
#ifdef ENABLE_STD_THREAD
    Utilities::writeMatrixToFile(stdThreadTimes, prefix + string("stdThreadTimes") + suffix);
#endif
    Utilities::writeMatrixToFile(ompStaticTimes, prefix + string("ompStaticTimes") + suffix);
    Utilities::writeMatrixToFile(ompDynamicTimes, prefix + string("ompDynamicTimes") + suffix);
    Utilities::writeMatrixToFile(ompGuidedTimes, prefix + string("ompGuidedTimes") + suffix);
    Utilities::writeMatrixToFile(tbbTimes, prefix + string("tbbTimes") + suffix);

  }
  printf("finished\n");

  return 0;
}
