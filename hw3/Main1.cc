// -*- C++ -*-
// Main1.cc
// cs181j hw3 Problem 1
// This is an exercise in measuring runtime and cache misses of an example
//  problem of matrix multiplication, with tiling

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

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

// Students have given feedback that they would like the functors
//  they're supposed to work on to be split into another file, so here
//  it is.
#include "Main1_functors.h"

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
  double maxError = std::abs(correctResult[0] - result[0]);
  unsigned int indexOfEntryWithMaxError = 0;
  unsigned int numberOfErrors = 0;
  for (unsigned int entryIndex = 0; entryIndex < result.size(); ++entryIndex) {
    const double error =
      std::abs(correctResult[entryIndex] - result[entryIndex]);
    if (error > maxError) {
      maxError = error;
      indexOfEntryWithMaxError = entryIndex;
    }
    if (error > absoluteErrorTolerance) {
      if (numberOfErrors < 10) {
        fprintf(stderr, "error in entry %6u: correct has %11.4e, test has "
                "%11.4e\n", entryIndex, correctResult[entryIndex],
                result[entryIndex]);
      }
      ++numberOfErrors;
    }
  }
  if (maxError > absoluteErrorTolerance) {
    sprintf(sprintfBuffer, "different results detected: "
            "the largest difference in an entry is entry %u error %11.4e, "
            "method named " BOLD_ON FG_RED "%s" RESET "\n",
            indexOfEntryWithMaxError, maxError, methodName.c_str());
    throw std::runtime_error(sprintfBuffer);
  }
}

template <class Multiplier>
void
runTimingTest(const unsigned int numberOfTrials,
              const int papiEventSet,
              const Multiplier multiplier,
              const vector<double> & leftMatrix,
              const vector<double> & rightMatrix,
              vector<double> * resultMatrix,
              double * elapsedTime,
              long long * numberOfL1CacheMisses) {

  long long papiCounters[1];
  *elapsedTime = std::numeric_limits<double>::max();
  *numberOfL1CacheMisses = std::numeric_limits<long long>::max();

  for (unsigned int trialNumber = 0;
       trialNumber < numberOfTrials; ++trialNumber) {

    // Reset the result matrix
    std::fill(resultMatrix->begin(), resultMatrix->end(), 0.);

    // Clear the cache
    Utilities::clearCpuCache();

    // Start measuring
    handlePapiError(PAPI_accum(papiEventSet, &papiCounters[0]));
    const high_resolution_clock::time_point tic = high_resolution_clock::now();

    // Do the multiplication
    multiplier.multiplyMatrices(leftMatrix, rightMatrix, resultMatrix);

    // Stop measuring
    const high_resolution_clock::time_point toc = high_resolution_clock::now();
    papiCounters[0] = 0;
    handlePapiError(PAPI_accum(papiEventSet, papiCounters));
    const double thisTrialsElapsedTime =
      duration_cast<duration<double> >(toc - tic).count();
    const long long thisTrialsNumberOfL1CacheMisses = papiCounters[0];
    // Take the minimum values from all trials
    *elapsedTime = std::min(*elapsedTime, thisTrialsElapsedTime);
    *numberOfL1CacheMisses =
      std::min(*numberOfL1CacheMisses, thisTrialsNumberOfL1CacheMisses);
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
  // It's only a desired number because some combinations of user input may
  //  result in repeated values.
  const unsigned int desiredNumberOfDataPoints = 15;
  // You can choose to make your matrix multiplication method general
  //  (such that it works on matrices that are not just multiples of
  //  the tile size) or not, and this option controls whether you make
  //  matrices that are a multiple of the tile size.
  const MatrixSizeStyle matrixSizeStyle = MatrixSizesAreArbitrary;
  //const MatrixSizeStyle matrixSizeStyle = MatrixSizesAreMultiplesOfTheTileSize;
  // This hold the different tile sizes to be tested.
  const vector<unsigned int> tileSizes = {12, 25, 75, 100};

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
       dataPointIndex < desiredNumberOfDataPoints; ++dataPointIndex) {
    const size_t matrixSize =
      Utilities::interpolateNumberLinearlyOnLogScale(matrixSizeExtrema[0],
                                                     matrixSizeExtrema[1],
                                                     desiredNumberOfDataPoints,
                                                     dataPointIndex);
    matrixSizes.push_back(matrixSize);
  }
  // Because some combinations of user input can make duplicate matrix sizes,
  //  we make them unique.
  std::sort(matrixSizes.begin(), matrixSizes.end());
  const std::vector<unsigned int>::iterator lastUnique =
    std::unique(matrixSizes.begin(), matrixSizes.end());
  matrixSizes.erase(lastUnique, matrixSizes.end());

  // This prefix and suffix will determine where files will be written and
  //  their names.
  const string prefix = "data/Main1_";
  const string suffix = "_shuffler";

  // ===========================================================================
  // *************************** < Papi initialization> ************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  PAPI_library_init(PAPI_VER_CURRENT);
  int papiEventSet = PAPI_NULL;
  handlePapiError(PAPI_create_eventset(&papiEventSet));

  // Add level 1 cache misses recording.
  handlePapiError(PAPI_add_event(papiEventSet, PAPI_L1_TCM));

  // Start the event set.
  handlePapiError(PAPI_start(papiEventSet));

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // *************************** </Papi initialization> ************************
  // ===========================================================================

  // Open the file that will store our data.
  char sprintfBuffer[500];
  sprintf(sprintfBuffer, "%sdata%s.csv", prefix.c_str(), suffix.c_str());
  FILE* file = fopen(sprintfBuffer, "w");

  // Here we write out the csv headers for our three techniques.
  fprintf(file, "matrixSize, trials");
  fprintf(file, ", ColRowLevel1Misses, ColRowTime");
  fprintf(file, ", RowColLevel1Misses, RowColTime");
  fprintf(file, ", improvedRowColLevel1Misses, improvedRowColTime");
  for (const unsigned int tileSize : tileSizes) {
    ignoreUnusedVariable(tileSize);
    fprintf(file, ", tileSize, tiledMatrixSize");
    fprintf(file, ", TiledMultiplication_Level1Misses, "
            "TiledMultiplication_Time");
    fprintf(file, ", TiledMultiplication_Improved_Level1Misses, "
            "TiledMultiplication_Improved_Time");
  }
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
    // The following calculates a number of trials for this size by trying
    //  to have the same number of flops performed, regardless of the matrix
    //  size.
    // It also has a min and max, for sanity.
    const double analyticFlopsForThisSize = 2 * std::pow(matrixSize, 3);
    const double targetNumberOfFlopsPerSize = 1e8;
    const unsigned int minimumNumberOfTrials = 3;
    const unsigned int maximumNumberOfTrials = 10;
    const unsigned int numberOfTrials =
      std::max(minimumNumberOfTrials,
               std::min(maximumNumberOfTrials,
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
    fprintf(file, "%6u, %10.6e", matrixSize, double(numberOfTrials));

    double minElapsedTime = std::numeric_limits<double>::max();
    long long minNumberOfLevel1CacheMisses =
      std::numeric_limits<long long>::max();

    // Do column major by row major
    {
      typedef MatrixMultiplier_ColMajorByRowMajor Multiplier;
      const Multiplier multiplier(matrixSize);
      runTimingTest<Multiplier>(numberOfTrials,
                                papiEventSet,
                                multiplier,
                                colMajorLeftMatrix,
                                rowMajorRightMatrix,
                                &rowMajorResultMatrix,
                                &minElapsedTime,
                                &minNumberOfLevel1CacheMisses);
      // Write output
      fprintf(file, ", %10.6e, %10.6e",
              double(minNumberOfLevel1CacheMisses),
              minElapsedTime);
    }
    // Store the "right" result to compare with all others
    const vector<double> correctRowMajorResultMatrix =
      rowMajorResultMatrix;

    {
      // Do row major by column major
      typedef MatrixMultiplier_RowMajorByColMajor Multiplier;
      const Multiplier multiplier(matrixSize);
      runTimingTest<Multiplier>(numberOfTrials,
                                papiEventSet,
                                multiplier,
                                rowMajorLeftMatrix,
                                colMajorRightMatrix,
                                &rowMajorResultMatrix,
                                &minElapsedTime,
                                &minNumberOfLevel1CacheMisses);
      // Check the result
      checkResult(correctRowMajorResultMatrix,
                  rowMajorResultMatrix,
                  absoluteErrorTolerance,
                  string("rowCol"));
      // Write output
      fprintf(file, ", %10.6e, %10.6e",
              double(minNumberOfLevel1CacheMisses),
              minElapsedTime);
    }

    {
      // Do improved row major by column major
      typedef MatrixMultiplier_RowMajorByColMajor_Improved Multiplier;
      const Multiplier multiplier(matrixSize);
      runTimingTest<Multiplier>(numberOfTrials,
                                papiEventSet,
                                multiplier,
                                rowMajorLeftMatrix,
                                colMajorRightMatrix,
                                &rowMajorResultMatrix,
                                &minElapsedTime,
                                &minNumberOfLevel1CacheMisses);
      // Check the result
      checkResult(correctRowMajorResultMatrix,
                  rowMajorResultMatrix,
                  absoluteErrorTolerance,
                  string("improved rowCol"));
      // Write output
      fprintf(file, ", %10.6e, %10.6e",
              double(minNumberOfLevel1CacheMisses),
              minElapsedTime);
    }


    // Now, we do the tiled version for each tile size
    for (const unsigned int tileSize : tileSizes) {

      // Determine the matrix size.
      // If sizes can be arbitrary, use the general size.  If not,
      //  round to the nearest multiple of the tile size.
      const unsigned int tiledMatrixSize =
        (matrixSizeStyle == MatrixSizesAreArbitrary) ?
        matrixSize :
        tileSize * std::max(unsigned(1),
                            unsigned(std::round(matrixSize / float(tileSize))));
      // Write output
      fprintf(file, ", %3u, %4u", tileSize, tiledMatrixSize);

      // Create new matrices, because we may be using a different size.
      const unsigned int numberOfTiledEntries =
        tiledMatrixSize * tiledMatrixSize;
      vector<double> rowMajorTiledLeftMatrix(numberOfTiledEntries);
      vector<double> rowMajorTiledRightMatrix(numberOfTiledEntries);
      vector<double> colMajorTiledRightMatrix(numberOfTiledEntries);
      for (unsigned int row = 0; row < tiledMatrixSize; ++row) {
        for (unsigned int col = 0; col < tiledMatrixSize; ++col) {
          rowMajorTiledLeftMatrix[row * tiledMatrixSize + col] =
            randomNumberGenerator(randomNumberEngine);
          const double rightRandomNumber =
            randomNumberGenerator(randomNumberEngine);
          rowMajorTiledRightMatrix[row * tiledMatrixSize + col] =
            rightRandomNumber;
          colMajorTiledRightMatrix[row + col * tiledMatrixSize] =
            rightRandomNumber;
        }
      }
      vector<double> rowMajorTiledResultMatrix(numberOfTiledEntries);

      // Do a serial calculation to determine the "right" answer
      {
        typedef MatrixMultiplier_RowMajorByColMajor_Improved Multiplier;
        const Multiplier multiplier(tiledMatrixSize);
        multiplier.multiplyMatrices(rowMajorTiledLeftMatrix,
                                    colMajorTiledRightMatrix,
                                    &rowMajorTiledResultMatrix);
      }
      // Store the "right" answer
      const vector<double> correctTiledResultMatrix = rowMajorTiledResultMatrix;


      {
        // Do tiled row major by row major
        typedef MatrixMultiplier_RowMajorByRowMajor_Tiled Multiplier;
        const Multiplier multiplier(tiledMatrixSize, tileSize);
        runTimingTest<Multiplier>(numberOfTrials,
                                  papiEventSet,
                                  multiplier,
                                  rowMajorTiledLeftMatrix,
                                  rowMajorTiledRightMatrix,
                                  &rowMajorTiledResultMatrix,
                                  &minElapsedTime,
                                  &minNumberOfLevel1CacheMisses);
        // Check the result
        checkResult(correctTiledResultMatrix,
                    rowMajorTiledResultMatrix,
                    absoluteErrorTolerance,
                    string("tiled"));
        // Write output
        fprintf(file, ", %10.6e, %10.6e",
                double(minNumberOfLevel1CacheMisses), minElapsedTime);
      }

    }

    // Output a heartbeat message
    const high_resolution_clock::time_point toc = high_resolution_clock::now();
    const double thisSizesElapsedTime =
      duration_cast<duration<double> >(toc - thisSizesTic).count();
    printf("finished size %8.2e (%3u/%3zu) with %8.2e trials in %8.2e seconds\n",
           float(matrixSize), matrixSizeIndex, matrixSizes.size(),
           float(numberOfTrials), thisSizesElapsedTime);

    fprintf(file, "\n");
    fflush(file);
  }
  fclose(file);

  // Clean up papi
  handlePapiError(PAPI_stop(papiEventSet, 0));
  handlePapiError(PAPI_cleanup_eventset(papiEventSet));
  handlePapiError(PAPI_destroy_eventset(&papiEventSet));

  return 0;
}
