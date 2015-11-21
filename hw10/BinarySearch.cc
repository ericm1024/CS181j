// -*- C++ -*-
// BinarySearch.cc
// cs181j hw10
// In this example we do binary searches on the cpu and gpu for comparison.

// These utilities are used on many assignments
#include "../Utilities.h"

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

#include <set>

#include "BinarySearch_cuda.cuh"

using std::string;
using std::vector;
using std::array;
using std::size_t;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

bool
findKeyInSortedNumbers_cpu(const unsigned int * const sortedNumbers,
                           const unsigned int numberOfSortedNumbers,
                           const unsigned int key) {
  unsigned int minIndex = 0;
  unsigned int maxIndex = numberOfSortedNumbers;
  if (key > sortedNumbers[numberOfSortedNumbers - 1]) {
    return false;
  }
  if (key < sortedNumbers[0]) {
    return false;
  }

  while (maxIndex >= minIndex) {
    const unsigned int midpointIndex = (minIndex + maxIndex) / 2;
    if (midpointIndex >= numberOfSortedNumbers) {
      printf("invalid midpoint index %4u for key %5u\n",
             midpointIndex, key);
      exit(1);
    }
    if (sortedNumbers[midpointIndex] == key) {
      return true;
    } else if (sortedNumbers[midpointIndex] < key) {
      minIndex = midpointIndex + 1;
    } else {
      maxIndex = midpointIndex - 1;
    }
  }
  return false;
}

bool
findKeyInSortedNumbers_stl(const unsigned int * const sortedNumbers,
                           const unsigned int numberOfSortedNumbers,
                           const unsigned int key) {
  return std::binary_search(&sortedNumbers[0],
                            &sortedNumbers[numberOfSortedNumbers],
                            key);
}

void
checkResult(const bool * const correctResult,
            const bool * const testResult,
            const unsigned int inputSize,
            const string & testName) {
  char sprintfBuffer[500];
  for (size_t i = 0; i < inputSize; ++i) {
    if (correctResult[i] != testResult[i]) {
      sprintf(sprintfBuffer, "wrong value for entry number %zu in test result, "
              "it's %s but should be %s, test named "
              BOLD_ON FG_RED "%s" RESET "\n", i,
              (testResult[i] == true) ? "true" : "false",
              (correctResult[i] == true) ? "true" : "false",
              testName.c_str());
      throw std::runtime_error(sprintfBuffer);
    }
  }
}

template <class Function>
void
runCpuTimingTest(const unsigned int numberOfTrials,
                 const vector<unsigned int> & sortedNumbers,
                 const vector<unsigned int> & input,
                 Function function,
                 bool * output,
                 double * elapsedTime) {

  *elapsedTime = std::numeric_limits<double>::max();

  for (unsigned int trialNumber = 0;
       trialNumber < numberOfTrials; ++trialNumber) {

    // Reset the cpu's cache
    Utilities::clearCpuCache();

    // Start measuring
    const high_resolution_clock::time_point tic = high_resolution_clock::now();

    // Do the searches
    const unsigned int numberOfSortedNumbers = sortedNumbers.size();
    const unsigned int inputSize = input.size();
    for (unsigned int index = 0; index < inputSize; ++index) {
      output[index] =
        function(&sortedNumbers[0],
                 numberOfSortedNumbers,
                 input[index]);
    }

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
  // This controls how many inputs are searched for.
  const array<double, 2> & rangeOfNumberOfInputs = {{1e2, 1e6}};
  // This number controls how many data points are made and plotted.
  const unsigned int numberOfInputSizeDataPoints = 9;
  // This controls how large the table size is (the number of sorted numbers)
  const array<double, 2> & rangeOfTableSizes = {{1e1, 1e4}};
  // This number controls how many data points are made and plotted.
  const unsigned int numberOfTableSizeDataPoints = 10;
  // This is the standard number of times the calculation is repeated.
  const unsigned int numberOfTrials = 5;
  const unsigned int numberOfThreadsPerBlock = 256;
  const unsigned int maxNumberOfBlocks = 1e4;
  const double successfulFindRate = 0.5;

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // *************************** </Inputs> *************************************
  // ===========================================================================

  const string prefix = "data/BinarySearch_";
  const string suffix = "_shuffler";

  std::mt19937 randomNumberEngine;
  std::uniform_int_distribution<unsigned int> randomNumberGenerator(0, std::numeric_limits<unsigned int>::max());

  vector<vector<double> >
    inputSizeMatrixForPlotting(numberOfInputSizeDataPoints,
                               vector<double>(numberOfTableSizeDataPoints));
  vector<vector<double> >
    tableSizeMatrixForPlotting(numberOfInputSizeDataPoints,
                               vector<double>(numberOfTableSizeDataPoints));
  vector<vector<double> >
    cpuTimes(numberOfInputSizeDataPoints,
             vector<double>(numberOfTableSizeDataPoints));
  vector<vector<double> >
    stlTimes(numberOfInputSizeDataPoints,
             vector<double>(numberOfTableSizeDataPoints));
  vector<vector<double> >
    gpuTimes(numberOfInputSizeDataPoints,
             vector<double>(numberOfTableSizeDataPoints));

  // for each size
  for (unsigned int inputSizeDataPointIndex = 0;
       inputSizeDataPointIndex < numberOfInputSizeDataPoints;
       ++inputSizeDataPointIndex) {
    // calculate this input size
    const size_t inputSize =
      Utilities::interpolateNumberLinearlyOnLogScale(rangeOfNumberOfInputs[0],
                                                     rangeOfNumberOfInputs[1],
                                                     numberOfInputSizeDataPoints,
                                                     inputSizeDataPointIndex);

    // generate the input, which are the numbers we'll be searching for
    std::set<unsigned int> inputSet;
    while (inputSet.size() != inputSize) {
      inputSet.insert(randomNumberGenerator(randomNumberEngine));
    }
    vector<unsigned int> input;
    std::copy(inputSet.begin(), inputSet.end(), std::back_inserter(input));
    std::random_shuffle(input.begin(), input.end());

    const high_resolution_clock::time_point thisSizesTic =
      high_resolution_clock::now();

    for (unsigned int tableSizeDataPointIndex = 0;
         tableSizeDataPointIndex < numberOfTableSizeDataPoints;
         ++tableSizeDataPointIndex) {
      // calculate this table size
      const size_t numberOfSortedNumbers =
        Utilities::interpolateNumberLinearlyOnLogScale(rangeOfTableSizes[0],
                                                       rangeOfTableSizes[1],
                                                       numberOfTableSizeDataPoints,
                                                       tableSizeDataPointIndex);

      // generate the sorted numbers
      vector<unsigned int> sortedNumbers = input;
      // shuffle the numbers
      std::random_shuffle(sortedNumbers.begin(), sortedNumbers.end());
      // chop off the ones we don't need
      sortedNumbers.resize(std::min(sortedNumbers.size(),
                                    size_t(numberOfSortedNumbers * successfulFindRate)));
      std::set<unsigned int> sortedSet;
      for (const unsigned u : sortedNumbers) {
        sortedSet.insert(u);
      }
      while (sortedSet.size() != numberOfSortedNumbers) {
        sortedSet.insert(randomNumberGenerator(randomNumberEngine));
      }
      sortedNumbers.resize(0);
      std::copy(sortedSet.begin(), sortedSet.end(), std::back_inserter(sortedNumbers));

      // each entry in output is a flag saying if the corresponding entry in the
      //  input was found in the sorted numbers
      bool * output = new bool[inputSize];
      std::fill(&output[0], &output[input.size()], false);

      // do the stl version
      runCpuTimingTest(numberOfTrials,
                       sortedNumbers,
                       input,
                       findKeyInSortedNumbers_stl,
                       output,
                       &stlTimes[inputSizeDataPointIndex][tableSizeDataPointIndex]);
      bool * correctOutput = new bool[inputSize];
      std::copy(&output[0], &output[inputSize], &correctOutput[0]);

      std::fill(&output[0], &output[inputSize], false);
      // do the cpu version
      runCpuTimingTest(numberOfTrials,
                       sortedNumbers,
                       input,
                       findKeyInSortedNumbers_cpu,
                       output,
                       &cpuTimes[inputSizeDataPointIndex][tableSizeDataPointIndex]);
      checkResult(correctOutput,
                  output,
                  inputSize,
                  std::string("cpu"));

      // fill output with false
      std::fill(&output[0], &output[input.size()], false);
      // do the gpu version
      runGpuTimingTest(numberOfTrials,
                       maxNumberOfBlocks,
                       numberOfThreadsPerBlock,
                       &sortedNumbers[0],
                       sortedNumbers.size(),
                       &input[0],
                       input.size(),
                       output,
                       &gpuTimes[inputSizeDataPointIndex][tableSizeDataPointIndex]);
      checkResult(correctOutput,
                  output,
                  inputSize,
                  std::string("gpu"));

      inputSizeMatrixForPlotting[inputSizeDataPointIndex][tableSizeDataPointIndex] =
        inputSize;
      tableSizeMatrixForPlotting[inputSizeDataPointIndex][tableSizeDataPointIndex] =
        numberOfSortedNumbers;

      delete[] output;
      delete[] correctOutput;
    }

    const high_resolution_clock::time_point thisSizesToc =
      high_resolution_clock::now();
    const double thisSizesElapsedTime =
      duration_cast<duration<double> >(thisSizesToc - thisSizesTic).count();
    printf("finished size %8.2e with %8.2e trials in %6.2f seconds\n",
           double(inputSize), double(numberOfTrials),
           thisSizesElapsedTime);
  }

  Utilities::writeMatrixToFile(inputSizeMatrixForPlotting,
                               prefix + string("inputSize") + suffix);
  Utilities::writeMatrixToFile(tableSizeMatrixForPlotting,
                               prefix + string("tableSize") + suffix);
  Utilities::writeMatrixToFile(stlTimes,
                               prefix + string("stl") + suffix);
  Utilities::writeMatrixToFile(cpuTimes,
                               prefix + string("cpu") + suffix);
  Utilities::writeMatrixToFile(gpuTimes,
                               prefix + string("gpu") + suffix);

  return 0;
}
