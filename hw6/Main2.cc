// -*- C++ -*-
// Main2.cc
// cs181j hw6 Problem 2
// An example to illustrate how to implement relatively simple
//  threading for a scalar function integrator.

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

#include "Main2_functions.h"

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
runTest(const unsigned int numberOfTrials,
        const Function function,
        const unsigned int numberOfThreads,
        const unsigned int numberOfIntervals,
        const array<double, 2> & integrationRange,
        double * const integral,
        double * const elapsedTime) {

  *elapsedTime = std::numeric_limits<double>::max();

  for (unsigned int trialNumber = 0;
       trialNumber < numberOfTrials; ++trialNumber) {

    // Start measuring
    const high_resolution_clock::time_point tic = high_resolution_clock::now();

    // Do the test
    *integral = function(numberOfThreads, numberOfIntervals, integrationRange);

    // Stop measuring
    const high_resolution_clock::time_point toc = high_resolution_clock::now();
    const double thisTrialsElapsedTime =
      duration_cast<duration<double> >(toc - tic).count();
    // Take the minimum values from all trials
    *elapsedTime = std::min(*elapsedTime, thisTrialsElapsedTime);
  }

}

// like my parents when i was in high school, we take no arguments
//int main(int argc, char* argv[]) {
int main() {

  // ===========================================================================
  // *************************** < Inputs> *************************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  const array<double, 2> numberOfIntervalsRange = {{50, 1e7}};
  const unsigned int numberOfDataPoints  = 20;
  // the integration bounds
  const array<double, 2> integrationRange = {{.61, 1.314}};
  const vector<unsigned int> numbersOfThreads =
    {{1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 36, 48}};

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // *************************** </Inputs> *************************************
  // ===========================================================================

  // On each test, we need to make sure we get the same result.  A test will
  //  fail if the difference between any entry in our result is more than
  //  relativeErrorTolerance different than entries we got with another method.
  const double relativeErrorTolerance = 1e-5;

  const string prefix = "data/Main2_";
  const string suffix = "_shuffler";

  // Make sure that the data directory exists.
  Utilities::verifyThatDirectoryExists("data");

  vector<vector<double> >
    numberOfIntervalsMatrixForPlotting(numberOfDataPoints,
                                       vector<double>(numbersOfThreads.size(), 0));
  vector<vector<double> >
    numberOfThreadsMatrixForPlotting(numberOfDataPoints,
                                     vector<double>(numbersOfThreads.size(), 0));
  vector<vector<double> >
    serialTimes(numberOfDataPoints,
                vector<double>(numbersOfThreads.size(), 0));
  vector<vector<double> >
    stdThreadTimes(numberOfDataPoints,
                   vector<double>(numbersOfThreads.size(), 0));
  vector<vector<double> >
    ompTimes(numberOfDataPoints,
             vector<double>(numbersOfThreads.size(), 0));

  // for each numberOfIntervals
  for (unsigned int dataPointIndex = 0;
       dataPointIndex < numberOfDataPoints;
       ++dataPointIndex) {

    const high_resolution_clock::time_point thisSizesTic =
      high_resolution_clock::now();

    const unsigned int numberOfIntervals =
      Utilities::interpolateNumberLinearlyOnLogScale(numberOfIntervalsRange[0],
                                                     numberOfIntervalsRange[1],
                                                     numberOfDataPoints,
                                                     dataPointIndex);

    const unsigned int maxNumberOfTrials = 100;
    const unsigned int minNumberOfTrials = 3;
    const unsigned int targetNumberOfIntervals = 1e7;
    const unsigned int numberOfTrials =
      std::min(maxNumberOfTrials,
               std::max(minNumberOfTrials,
                        targetNumberOfIntervals / numberOfIntervals));

    double serialIntegral =
      std::numeric_limits<double>::quiet_NaN();
    double serialElapsedTime =
      std::numeric_limits<double>::quiet_NaN();
    const unsigned int numberOfThreadsForSerial = 1;
    runTest(numberOfTrials,
            calculateIntegral_serial,
            numberOfThreadsForSerial, // this is really ignored
            numberOfIntervals,
            integrationRange,
            &serialIntegral,
            &serialElapsedTime);

    // for each numberOfThreads
    for (unsigned int numberOfThreadsIndex = 0;
         numberOfThreadsIndex < numbersOfThreads.size();
         ++numberOfThreadsIndex) {
      // calculate the number of threads
      const unsigned int numberOfThreads =
        numbersOfThreads[numberOfThreadsIndex];

      double integral =
        std::numeric_limits<double>::quiet_NaN();

      try {

        // set the serial time
        serialTimes[dataPointIndex][numberOfThreadsIndex] =
          serialElapsedTime;

        // std thread version
        runTest(numberOfTrials,
                calculateIntegral_stdThread,
                numberOfThreads,
                numberOfIntervals,
                integrationRange,
                &integral,
                &stdThreadTimes[dataPointIndex][numberOfThreadsIndex]);
        checkResult(serialIntegral,
                    integral,
                    string("std::thread"),
                    relativeErrorTolerance);

        // openmp version
        runTest(numberOfTrials,
                calculateIntegral_omp,
                numberOfThreads,
                numberOfIntervals,
                integrationRange,
                &integral,
                &ompTimes[dataPointIndex][numberOfThreadsIndex]);
        checkResult(serialIntegral,
                    integral,
                    string("openmp"),
                    relativeErrorTolerance);

      } catch (const std::exception & e) {
        fprintf(stderr, "error attempting %e intervals and %2u threads\n",
                float(numberOfIntervals), numberOfThreads);
        throw;
      }


      numberOfIntervalsMatrixForPlotting[dataPointIndex][numberOfThreadsIndex] =
        numberOfIntervals;
      numberOfThreadsMatrixForPlotting[dataPointIndex][numberOfThreadsIndex] =
        numberOfThreads;
    }

    const high_resolution_clock::time_point thisSizesToc =
      high_resolution_clock::now();
    const double thisSizesElapsedTime =
      duration_cast<duration<double> >(thisSizesToc - thisSizesTic).count();
    printf("finished size %8.2e with %8.2e trials in %6.2f seconds\n",
           double(numberOfIntervals), double(numberOfTrials),
           thisSizesElapsedTime);

  }

  Utilities::writeMatrixToFile(numberOfIntervalsMatrixForPlotting,
                               prefix + string("numberOfIntervals") + suffix);
  Utilities::writeMatrixToFile(numberOfThreadsMatrixForPlotting,
                               prefix + string("numberOfThreads") + suffix);
  Utilities::writeMatrixToFile(serialTimes, prefix + string("serial") + suffix);
  Utilities::writeMatrixToFile(stdThreadTimes, prefix + string("stdThread") + suffix);
  Utilities::writeMatrixToFile(ompTimes, prefix + string("omp") + suffix);

  return 0;
}
