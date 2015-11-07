// -*- C++ -*-
// Main3.cc
// cs181j hw8 Problem 3
// An example to illustrate how to implement threading for a histogram
//  calculator with reductions and locks, using omp and tbb

// These utilities are used on many assignments
#include "../Utilities.h"

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

#include "Main3_functions.h"

using std::string;
using std::vector;
using std::array;
using std::size_t;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

void
writeColumnToFile(const vector<double> & times,
                  const string & filename) {
  const string appendedFilename = filename + string(".csv");
  FILE* file = fopen(appendedFilename.c_str(), "w");
  for (unsigned int i = 0; i < times.size(); ++i) {
    fprintf(file, "%e\n", times[i]);
  }
  fclose(file);
  printf("wrote file to %s\n", appendedFilename.c_str());
}

void
checkResult(const vector<unsigned int> & correctResult,
            const vector<unsigned int> & testResult,
            const string & testName) {
  char sprintfBuffer[500];
  const unsigned int numberOfBuckets = correctResult.size();
  if (testResult.size() != numberOfBuckets) {
    sprintf(sprintfBuffer, "Error found: "
            "testResult.size() = %zu, but correctResult.size() = %zu, "
            "test named "
            BOLD_ON FG_RED "%s" RESET "\n",
            testResult.size(), correctResult.size(),
            testName.c_str());
    throw std::runtime_error(sprintfBuffer);
  }
  for (unsigned int bucketIndex = 0;
       bucketIndex < numberOfBuckets; ++bucketIndex) {
    if (testResult[bucketIndex] != correctResult[bucketIndex]) {
      sprintf(sprintfBuffer, "Error found: "
              "testResult[%u] = %u, but correctResult[%u] = %u, "
              "test named "
              BOLD_ON FG_RED "%s" RESET "\n",
              bucketIndex, testResult[bucketIndex],
              bucketIndex, correctResult[bucketIndex],
              testName.c_str());
      throw std::runtime_error(sprintfBuffer);
    }
  }
}

template < class Function>
void
runTimingTest(const unsigned int numberOfTrials,
              const Function function,
              const vector<double> & input,
              const unsigned int lockBucketSize,
              double * elapsedTime,
              vector<unsigned int> * histogram) {

  *elapsedTime = std::numeric_limits<double>::max();

  for (unsigned int trialNumber = 0;
       trialNumber < numberOfTrials; ++trialNumber) {

    // don't time the filling
    std::fill(histogram->begin(), histogram->end(), 0);

    // Start measuring
    const high_resolution_clock::time_point tic = high_resolution_clock::now();

    // Do the calculation
    function(input, lockBucketSize, histogram);

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
  const unsigned int nominalSize = 1e6;
  // modify the input size to be a multiple of 1024
  const unsigned int inputSize = unsigned(nominalSize / 1024) * 1024;
  const vector<unsigned int> numbersOfThreads =
    {{1, 2, 4, 8, 16, 32, 64}};
  // How many buckets in the histogram
  const unsigned int numberOfBuckets = 1024;
  // These are the lock bucket sizes.  A size of 1 means that each lock is in
  //  charge of a single bucket.  A size of 64 means that each lock is in charge
  //  of 64 buckets.
  const vector<unsigned int> lockBucketSizes =
    {{1, 2, 4, 8, 16, 32, 64}};
  const unsigned int numberOfTrials = 3;

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // *************************** </Inputs> *************************************
  // ===========================================================================

  // create a random number generator
  std::mt19937 randomNumberEngine;
  std::uniform_real_distribution<double> randomNumberGenerator(0, 1);

  vector<double> input(inputSize);
  // all values in input will be in the range of [0, 1)
  for(unsigned int index = 0; index < inputSize; ++index) {
    input[index] = randomNumberGenerator(randomNumberEngine);
  }

  const string prefix = "data/Main3_";
  const string suffix = "_shuffler";

  const unsigned int lockBucketSizeForNoLocks =
    std::numeric_limits<unsigned int>::max();

  vector<unsigned int> serialHistogram(numberOfBuckets, 0);
  double serialElapsedTime;
  runTimingTest(numberOfTrials,
                calculateHistogram_serial,
                input,
                lockBucketSizeForNoLocks,
                &serialElapsedTime,
                &serialHistogram);

  // lockfree versions
  vector<double> serialTimes(numbersOfThreads.size(), 0);
  vector<double> reductionTimes(numbersOfThreads.size(), 0);
  vector<double> atomicsTimes(numbersOfThreads.size(), 0);
  vector<double> tbbReductionTimes(numbersOfThreads.size(), 0);
  vector<double> tbbAtomicsTimes(numbersOfThreads.size(), 0);

  vector<vector<double> >
    numberOfThreadsMatrixForPlotting(numbersOfThreads.size(),
                                     vector<double>(lockBucketSizes.size(), 0));
  vector<vector<double> >
    lockBucketSizeMatrixForPlotting(numbersOfThreads.size(),
                                    vector<double>(lockBucketSizes.size(), 0));
  vector<vector<double> >
    serialMatrixTimes(numbersOfThreads.size(),
                      vector<double>(lockBucketSizes.size(), 0));
  vector<vector<double> >
    atomicFlagLocksTimes(numbersOfThreads.size(),
                         vector<double>(lockBucketSizes.size(), 0));
  vector<vector<double> >
    tbbLocksTimes(numbersOfThreads.size(),
                  vector<double>(lockBucketSizes.size(), 0));

  for (unsigned int numberOfThreadsIndex = 0;
       numberOfThreadsIndex < numbersOfThreads.size();
       ++numberOfThreadsIndex) {

    const high_resolution_clock::time_point thisSizesTic =
      high_resolution_clock::now();

    const unsigned int numberOfThreads =
      numbersOfThreads[numberOfThreadsIndex];

    // initialize threading systems for this number of threads
    omp_set_num_threads(numberOfThreads);
    tbb::task_scheduler_init init(numberOfThreads);

    // reduction
    vector<unsigned int> histogram(numberOfBuckets, 0);
    runTimingTest(numberOfTrials,
                  calculateHistogram_reduction,
                  input,
                  lockBucketSizeForNoLocks,
                  &reductionTimes[numberOfThreadsIndex],
                  &histogram);
    checkResult(serialHistogram, histogram, string("reduction"));

    // atomics
    runTimingTest(numberOfTrials,
                  calculateHistogram_atomics,
                  input,
                  lockBucketSizeForNoLocks,
                  &atomicsTimes[numberOfThreadsIndex],
                  &histogram);
    checkResult(serialHistogram, histogram, string("atomics"));

    // tbbReduction
    runTimingTest(numberOfTrials,
                  calculateHistogram_tbbReduction,
                  input,
                  lockBucketSizeForNoLocks,
                  &tbbReductionTimes[numberOfThreadsIndex],
                  &histogram);
    checkResult(serialHistogram, histogram, string("tbb reduction"));

    // tbbAtomics
    runTimingTest(numberOfTrials,
                  calculateHistogram_tbbAtomics,
                  input,
                  lockBucketSizeForNoLocks,
                  &tbbAtomicsTimes[numberOfThreadsIndex],
                  &histogram);
    checkResult(serialHistogram, histogram, string("tbb atomics"));

    serialTimes[numberOfThreadsIndex] = serialElapsedTime;


    // now, do the lock versions
    // for each lockBucketSize
    for (unsigned int lockBucketSizeIndex = 0;
         lockBucketSizeIndex < lockBucketSizes.size();
         ++lockBucketSizeIndex) {

      const unsigned int lockBucketSize =
        lockBucketSizes[lockBucketSizeIndex];

      // non-tbb locks
      runTimingTest(numberOfTrials,
                    calculateHistogram_atomicFlagLocks,
                    input,
                    lockBucketSize,
                    &atomicFlagLocksTimes[numberOfThreadsIndex][lockBucketSizeIndex],
                    &histogram);
      checkResult(serialHistogram, histogram, string("atomic flag locks"));

      // tbb locks
      runTimingTest(numberOfTrials,
                    calculateHistogram_tbbLocks,
                    input,
                    lockBucketSize,
                    &tbbLocksTimes[numberOfThreadsIndex][lockBucketSizeIndex],
                    &histogram);
      checkResult(serialHistogram, histogram, string("tbb locks"));

      serialMatrixTimes[numberOfThreadsIndex][lockBucketSizeIndex] = serialElapsedTime;
      lockBucketSizeMatrixForPlotting[numberOfThreadsIndex][lockBucketSizeIndex] =
        lockBucketSize;
      numberOfThreadsMatrixForPlotting[numberOfThreadsIndex][lockBucketSizeIndex] =
        numberOfThreads;
    }

    const high_resolution_clock::time_point thisSizesToc =
      high_resolution_clock::now();
    const double thisSizesElapsedTime =
      duration_cast<duration<double> >(thisSizesToc - thisSizesTic).count();
    printf("finished numberOfThreads %3u with %8.2e trials in %6.2f seconds\n",
           numberOfThreads, double(numberOfTrials),
           thisSizesElapsedTime);

  }

  writeColumnToFile(serialTimes,
                    prefix + string("serialTimes") + suffix);
  writeColumnToFile(reductionTimes,
                    prefix + string("reductionTimes") + suffix);
  writeColumnToFile(atomicsTimes,
                    prefix + string("atomicsTimes") + suffix);
  writeColumnToFile(tbbReductionTimes,
                    prefix + string("tbbReductionTimes") + suffix);
  writeColumnToFile(tbbAtomicsTimes,
                    prefix + string("tbbAtomicsTimes") + suffix);

  Utilities::writeMatrixToFile(lockBucketSizeMatrixForPlotting,
                               prefix + string("lockBucketSize") + suffix);
  Utilities::writeMatrixToFile(numberOfThreadsMatrixForPlotting,
                               prefix + string("numberOfThreads") + suffix);
  Utilities::writeMatrixToFile(serialMatrixTimes,
                               prefix + string("serialMatrixTimes") + suffix);
  Utilities::writeMatrixToFile(atomicFlagLocksTimes,
                               prefix + string("atomicFlagLocksTimes") + suffix);
  Utilities::writeMatrixToFile(tbbLocksTimes,
                               prefix + string("tbbLocksTimes") + suffix);
  return 0;
}
