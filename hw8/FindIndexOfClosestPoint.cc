// -*- C++ -*-
// FindIndexOfClosestPoint.cc
// cs181j hw8
// An example of threading a simple function, for warmup

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

#include "FindIndexOfClosestPoint_functions.h"

#include <tbb/task_scheduler_init.h>

using std::string;
using std::vector;
using std::array;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

// As usual, a result checking function to make sure we have the right answer.
void
checkResult(const unsigned int correctResult,
            const unsigned int testResult,
            const std::string & testName) {
  char sprintfBuffer[500];
  if (testResult != correctResult) {
    sprintf(sprintfBuffer, "wrong result, "
            "it's %u but should be %u, test named "
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
        const vector<Point> & points,
        const Point & searchLocation,
        unsigned int * const indexOfClosestPoint,
        double * const elapsedTime) {

  *elapsedTime = std::numeric_limits<double>::max();

  for (unsigned int trialNumber = 0;
       trialNumber < numberOfTrials; ++trialNumber) {

    // Start measuring
    const high_resolution_clock::time_point tic = high_resolution_clock::now();

    // Do the test
    *indexOfClosestPoint = function(numberOfThreads, points, searchLocation);

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

  const array<double, 2> numberOfPointsRange     = {{1e4, 1e7}};
  const unsigned int numberOfDataPoints          = 10;
  const vector<unsigned int> numbersOfThreads    =
    {{1, 2, 4, 6, 8, 10, 11, 12, 13, 14, 16, 20, 30, 40, 50}};
  const unsigned int numberOfTrials              = 20;

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // *************************** </Inputs> *************************************
  // ===========================================================================

  const string prefix = "data/FindIndexOfClosestPoint_";
  const string suffix = "_shuffler";

  // Make sure that the data directory exists.
  Utilities::verifyThatDirectoryExists("data");

  const unsigned randomSeed = 0;
  std::default_random_engine randomNumberEngine(randomSeed);
  std::uniform_real_distribution<double> randomNumberGenerator(0, 1);

  vector<vector<double> >
    numberOfPointsMatrixForPlotting(numberOfDataPoints,
                                    vector<double>(numbersOfThreads.size(), 0));
  vector<vector<double> >
    numberOfThreadsMatrixForPlotting(numberOfDataPoints,
                                     vector<double>(numbersOfThreads.size(), 0));
  vector<vector<double> >
    serialTimes(numberOfDataPoints,
                vector<double>(numbersOfThreads.size(), 0));
  vector<vector<double> >
    ompTimes(numberOfDataPoints,
             vector<double>(numbersOfThreads.size(), 0));
  vector<vector<double> >
    tbbTimes(numberOfDataPoints,
             vector<double>(numbersOfThreads.size(), 0));

  // for each numberOfPoints
  for (unsigned int dataPointIndex = 0;
       dataPointIndex < numberOfDataPoints;
       ++dataPointIndex) {

    const high_resolution_clock::time_point thisSizesTic =
      high_resolution_clock::now();

    const unsigned int numberOfPoints =
      Utilities::interpolateNumberLinearlyOnLogScale(numberOfPointsRange[0],
                                                     numberOfPointsRange[1],
                                                     numberOfDataPoints,
                                                     dataPointIndex);

    // make the points
    vector<Point> points(numberOfPoints);
    std::generate(points.begin(), points.end(), [&] () {
        return (Point) {{randomNumberGenerator(randomNumberEngine),
              randomNumberGenerator(randomNumberEngine),
              randomNumberGenerator(randomNumberEngine)}};
      });

    // make the search location
    const Point searchLocation =
      {{randomNumberGenerator(randomNumberEngine),
        randomNumberGenerator(randomNumberEngine),
        randomNumberGenerator(randomNumberEngine)}};

    unsigned int serialIndexOfClosestPoint;
    double serialElapsedTime =
      std::numeric_limits<double>::quiet_NaN();
    const unsigned int numberOfThreadsForSerial = 1;
    runTest(numberOfTrials,
            findIndexOfClosestPoint_serial,
            numberOfThreadsForSerial, // this is really ignored
            points,
            searchLocation,
            &serialIndexOfClosestPoint,
            &serialElapsedTime);

    // for each numberOfThreads
    for (unsigned int numberOfThreadsIndex = 0;
         numberOfThreadsIndex < numbersOfThreads.size();
         ++numberOfThreadsIndex) {
      // get the number of threads
      const unsigned int numberOfThreads =
        numbersOfThreads[numberOfThreadsIndex];

      // set the number of threads
      tbb::task_scheduler_init init(numberOfThreads);

      try {

        // set the serial time
        serialTimes[dataPointIndex][numberOfThreadsIndex] =
          serialElapsedTime;

        // threaded version
        unsigned int threadedIndexOfClosestPoint;
        runTest(numberOfTrials,
                findIndexOfClosestPoint_threaded,
                numberOfThreads,
                points,
                searchLocation,
                &threadedIndexOfClosestPoint,
                &ompTimes[dataPointIndex][numberOfThreadsIndex]);
        checkResult(serialIndexOfClosestPoint,
                    threadedIndexOfClosestPoint,
                    string("threaded"));

        // tbb version
        unsigned int tbbIndexOfClosestPoint;
        runTest(numberOfTrials,
                findIndexOfClosestPoint_tbb,
                numberOfThreads,
                points,
                searchLocation,
                &tbbIndexOfClosestPoint,
                &tbbTimes[dataPointIndex][numberOfThreadsIndex]);
        checkResult(serialIndexOfClosestPoint,
                    tbbIndexOfClosestPoint,
                    string("tbb"));

      } catch (const std::exception & e) {
        fprintf(stderr, "error attempting %e points and %2u threads\n",
                float(numberOfPoints), numberOfThreads);
        throw;
      }

      numberOfPointsMatrixForPlotting[dataPointIndex][numberOfThreadsIndex] =
        numberOfPoints;
      numberOfThreadsMatrixForPlotting[dataPointIndex][numberOfThreadsIndex] =
        numberOfThreads;
    }

    const high_resolution_clock::time_point thisSizesToc =
      high_resolution_clock::now();
    const double thisSizesElapsedTime =
      duration_cast<duration<double> >(thisSizesToc - thisSizesTic).count();
    printf("finished size %8.2e with %8.2e trials in %6.2f seconds\n",
           double(numberOfPoints), double(numberOfTrials),
           thisSizesElapsedTime);

  }

  Utilities::writeMatrixToFile(numberOfPointsMatrixForPlotting,
                               prefix + string("numberOfPoints") + suffix);
  Utilities::writeMatrixToFile(numberOfThreadsMatrixForPlotting,
                               prefix + string("numberOfThreads") + suffix);
  Utilities::writeMatrixToFile(serialTimes, prefix + string("serial") + suffix);
  Utilities::writeMatrixToFile(ompTimes, prefix + string("omp") + suffix);
  Utilities::writeMatrixToFile(tbbTimes, prefix + string("tbb") + suffix);

  return 0;
}
