// -*- C++ -*-
// Main2.cc
// cs181j hw1 Problem 2
// This is an exercise in measuring runtime and cache misses of an example
//  problem of matrix multiplication.

// Many of the homework assignments have definitions and includes that
//  are common across several executables, so we group them together.
#include "CommonDefinitions.h"

// Students have given feedback that they would like the functions
//  (later functors) they're supposed to work on to be split into
//  another file, so here it is.
#include "Main2_functions.h"

// Only bring in from the standard namespace things that we care about.
// Remember, it's a naughty thing to just use the whole namespace.
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::vector;
using std::string;
using std::array;

// All the magic for measuring cache misses is in papi.
#include <papi.h>

// A very thin wrapper around PAPI_strerror.
void
handlePapiError(int papiReturnVal) {
  if (papiReturnVal != PAPI_OK) {
    fprintf(stderr, "PAPI error: %s\n",
            PAPI_strerror(papiReturnVal));
    exit(1);
  }
}

// This utility checks the maximum difference between the entries in
//  a correct result and a test result.  If the difference is too large,
//  it throws an exception.
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

int main() {

  // ===========================================================================
  // *************************** < Inputs> *************************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  // A lot of homeworks will run something over a range of sizes,
  //  which will then be plotted by some script.  Here, we run from a
  //  matrix size of 25 to 600.
  const array<double, 2> matrixSizeExtrema = {{25, 600}};
  // This number controls how many matrix sizes are used.
  const unsigned int numberOfDataPoints = 10;

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // *************************** </Inputs> *************************************
  // ===========================================================================

  // On each test, we need to make sure we get the same result.  A test will
  //  fail if the difference between any entry in our result is more than
  //  absoluteErrorTolerance different than entries we got with another method.
  const double absoluteErrorTolerance = 1e-4;

  // Make sure that the data directory exists.
  Utilities::verifyThatDirectoryExists("data");

  // Generate the matrix sizes to test.
  vector<unsigned int> matrixSizes;
  for (unsigned int dataPointIndex = 0;
       dataPointIndex < numberOfDataPoints; ++dataPointIndex) {
    const size_t matrixSize =
      Utilities::interpolateNumberLinearlyOnLogScale(matrixSizeExtrema[0],
                                                     matrixSizeExtrema[1],
                                                     numberOfDataPoints,
                                                     dataPointIndex);
    matrixSizes.push_back(matrixSize);
  }
  // Because some combinations of user input can make duplicate matrix sizes,
  //  we make them unique.
  std::unique(matrixSizes.begin(), matrixSizes.end());

  // This prefix and suffix will determine where files will be written and
  //  their names.
  const string prefix = "data/Main2_";
  const string suffix = "_shuffler";

  // ===========================================================================
  // *************************** < Papi initialization> ************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  // TODO: initialize papi

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // *************************** </Papi initialization> ************************
  // ===========================================================================

  // Open the file that will store our data.
  char sprintfBuffer[500];
  sprintf(sprintfBuffer, "%sdata%s.csv", prefix.c_str(), suffix.c_str());
  FILE* file = fopen(sprintfBuffer, "w");

  // Here we write out the csv headers for our three techniques.
  fprintf(file, "size, repeats, analyticFlops");
  fprintf(file, ", ColRowLevel1Misses, ColRowTime");
  fprintf(file, ", RowColLevel1Misses, RowColTime");
  fprintf(file, ", improvedRowColLevel1Misses, improvedRowColTime");
  fprintf(file, "\n");

  // Create a random number generator
  std::mt19937 randomNumberEngine;
  std::uniform_real_distribution<double> randomNumberGenerator(0, 1);

  // For each matrix size
  for (unsigned int matrixSizeIndex = 0;
       matrixSizeIndex < matrixSizes.size(); ++matrixSizeIndex) {

    // Start timing, simply for a heartbeat message
    const high_resolution_clock::time_point thisSizesTic =
      high_resolution_clock::now();

    const unsigned int matrixSize = matrixSizes[matrixSizeIndex];
    const unsigned int numberOfEntries = matrixSize * matrixSize;

    // We need to repeat each test multiple times to improve our results.
    // We'd like to repeat small sizes many more times than large sizes
    //  because they're more noisy.
    // The following calculates a number of repeats for this size by trying
    //  to have the same number of flops performed, regardless of the matrix
    //  size.
    // It also has a min and max, for sanity.
    const double analyticFlopsForThisSize = 2 * std::pow(matrixSize, 3);
    const double targetNumberOfFlopsPerSize = 1e8;
    const unsigned int minimumNumberOfRepeats = 3;
    const unsigned int maximumNumberOfRepeats = 100;
    const unsigned int numberOfRepeats =
      std::max(minimumNumberOfRepeats,
               std::min(maximumNumberOfRepeats,
                        unsigned(targetNumberOfFlopsPerSize /
                                 analyticFlopsForThisSize)));

    // Make the matrices that we're multiplying.
    vector<double> colMajorLeftMatrix(numberOfEntries);
    vector<double> rowMajorLeftMatrix(numberOfEntries);
    vector<double> colMajorRightMatrix(numberOfEntries);
    vector<double> rowMajorRightMatrix(numberOfEntries);
    for (unsigned int row = 0; row < matrixSize; ++row) {
      for (unsigned int col = 0; col < matrixSize; ++col) {
        const double leftRandomNumber =
          randomNumberGenerator(randomNumberEngine);
        colMajorLeftMatrix[row + col * matrixSize] = leftRandomNumber;
        rowMajorLeftMatrix[row * matrixSize + col] = leftRandomNumber;
        const double rightRandomNumber =
          randomNumberGenerator(randomNumberEngine);
        colMajorRightMatrix[row + col * matrixSize] = rightRandomNumber;
        rowMajorRightMatrix[row * matrixSize + col] = rightRandomNumber;
      }
    }
    vector<double> rowMajorResultMatrix(numberOfEntries);

    // Write out some information for this size.
    fprintf(file, "%6u, %10.6e, %10.6e", matrixSize, double(numberOfRepeats),
            analyticFlopsForThisSize);

    double minElapsedTime = std::numeric_limits<double>::max();
    // TODO: use this variable like minElapsedTime to record and
    //  output the number of cache misses.
    long long minNumberOfLevel1CacheMisses =
      std::numeric_limits<long long>::max();

    // Now we'll actually do the colRow multiplications.
    // For each repeat
    for (unsigned int repeatNumber = 0;
         repeatNumber < numberOfRepeats; ++repeatNumber) {

      // Clear the cache
      Utilities::clearCpuCache();

      // Start measuring
      const high_resolution_clock::time_point tic = high_resolution_clock::now();

      Main2::multiplyColMajorByRowMajorMatrices(matrixSize,
                                                colMajorLeftMatrix,
                                                rowMajorRightMatrix,
                                                &rowMajorResultMatrix);

      // Stop measuring
      const high_resolution_clock::time_point toc = high_resolution_clock::now();
      const double thisRepeatsElapsedTime =
        duration_cast<duration<double> >(toc - tic).count();
      minElapsedTime = std::min(minElapsedTime, thisRepeatsElapsedTime);

    }
    // Store the "right" result to compare with all others
    const vector<double> correctRowMajorResultMatrix =
      rowMajorResultMatrix;
    // Write output
    fprintf(file, ", %10.6e, %10.6e",
            double(minNumberOfLevel1CacheMisses),
            minElapsedTime);



    // Reset the result
    std::fill(rowMajorResultMatrix.begin(), rowMajorResultMatrix.end(),
              std::numeric_limits<double>::quiet_NaN());
    // Set one value, to trigger the correctness test if nothing is implemented.
    rowMajorResultMatrix[0] = 1.;
    // Reset the counters
    minElapsedTime = std::numeric_limits<double>::max();
    minNumberOfLevel1CacheMisses =
      std::numeric_limits<long long>::max();



    // Now we'll do the rowCol multiplications
    for (unsigned int repeatNumber = 0;
         repeatNumber < numberOfRepeats; ++repeatNumber) {

      // Clear the cache
      Utilities::clearCpuCache();

      // Start measuring
      const high_resolution_clock::time_point tic = high_resolution_clock::now();

      Main2::multiplyRowMajorByColMajorMatrices(matrixSize,
                                                rowMajorLeftMatrix,
                                                colMajorRightMatrix,
                                                &rowMajorResultMatrix);

      // Stop measuring
      const high_resolution_clock::time_point toc = high_resolution_clock::now();
      const double thisRepeatsElapsedTime =
        duration_cast<duration<double> >(toc - tic).count();
      minElapsedTime = std::min(minElapsedTime, thisRepeatsElapsedTime);

    }
    // Check the result
    checkResult(correctRowMajorResultMatrix,
                rowMajorResultMatrix,
                absoluteErrorTolerance,
                string("rowCol"));
    // Write output
    fprintf(file, ", %10.6e, %10.6e",
            double(minNumberOfLevel1CacheMisses),
            minElapsedTime);



    // Reset the result
    std::fill(rowMajorResultMatrix.begin(), rowMajorResultMatrix.end(),
              std::numeric_limits<double>::quiet_NaN());
    // Set one value, to trigger the correctness test if nothing is implemented.
    rowMajorResultMatrix[0] = 1.;
    // Reset the counters
    minElapsedTime = std::numeric_limits<double>::max();
    minNumberOfLevel1CacheMisses =
      std::numeric_limits<long long>::max();



    // Now we'll do the improved rowCol multiplications
    for (unsigned int repeatNumber = 0;
         repeatNumber < numberOfRepeats; ++repeatNumber) {

      // Clear the cache
      Utilities::clearCpuCache();

      // Start measuring
      const high_resolution_clock::time_point tic = high_resolution_clock::now();

      Main2::multiplyRowMajorByColMajorMatrices_improved(matrixSize,
                                                         rowMajorLeftMatrix,
                                                         colMajorRightMatrix,
                                                         &rowMajorResultMatrix);

      // Stop measuring
      const high_resolution_clock::time_point toc = high_resolution_clock::now();
      const double thisRepeatsElapsedTime =
        duration_cast<duration<double> >(toc - tic).count();
      minElapsedTime = std::min(minElapsedTime, thisRepeatsElapsedTime);

    }
    // Check the result
    checkResult(correctRowMajorResultMatrix,
                rowMajorResultMatrix,
                absoluteErrorTolerance,
                string("improved rowCol"));
    // Write output
    fprintf(file, ", %10.6e, %10.6e",
            double(minNumberOfLevel1CacheMisses),
            minElapsedTime);



    // Output a heartbeat message
    const high_resolution_clock::time_point toc = high_resolution_clock::now();
    const double thisSizesElapsedTime =
      duration_cast<duration<double> >(toc - thisSizesTic).count();
    printf("finished size %8.2e (%3u/%3zu) with %8.2e repeats in %8.2e seconds\n",
           float(matrixSize), matrixSizeIndex, matrixSizes.size(),
           float(numberOfRepeats), thisSizesElapsedTime);

    fprintf(file, "\n");
    fflush(file);
  }
  fclose(file);

  // TODO: Clean up papi

  return 0;
}
