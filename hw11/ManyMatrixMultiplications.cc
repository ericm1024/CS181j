// -*- C++ -*-
// ManyMatrixMultiplications.cc
// cs181j hw11
// In this exercise, we do many matrix multiplications on the cpu and gpu

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

// These utilities are used on many assignments
#include "../Utilities.h"

using std::string;
using std::vector;
using std::array;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

#include "ManyMatrixMultiplications_cuda.cuh"

void
checkResult(const vector<double> & correctResult,
            const vector<double> & testResult,
            const unsigned int numberOfMatricesToMultiply,
            const unsigned int matrixSize,
            const string & testName) {
  char sprintfBuffer[500];
  for (unsigned int matrixIndex = 0;
       matrixIndex < numberOfMatricesToMultiply; ++matrixIndex) {
    for (unsigned int row = 0; row < matrixSize; ++row) {
      for (unsigned int col = 0; col < matrixSize; ++col) {
        const double correctNumber =
          correctResult[matrixIndex * matrixSize * matrixSize +
                        row * matrixSize + col];
        const double ourNumber =
          testResult[matrixIndex * matrixSize * matrixSize +
                     row * matrixSize + col];
        if (std::isfinite(ourNumber) == false ||
            std::abs(correctNumber - ourNumber) > 1e-4) {
          sprintf(sprintfBuffer, "incorrect entry in matrix %u row %u col %u, "
                  "correct number is %e, our number is %e, flavor is %s\n",
                  matrixIndex, row, col, correctNumber, ourNumber,
                  testName.c_str());
          throw std::runtime_error(sprintfBuffer);
        }
      }
    }
  }
}

string
convertCudaManyMatrixMultiplicationStyleToString(const CudaManyMatrixMultiplicationStyle cudaStyle) {
  switch (cudaStyle) {
  case NextThreadNextEntry_serialMatrices:
    return string("nextEntry_serialMatrices");
  case NextThreadNextMatrix_deepEntries:
    return string("nextMatrix_deepEntries");
  default:
    fprintf(stderr, "invalid cuda many matrix multiplication style\n");
    exit(1);
  };
}

template <class Function>
void
runCpuTimingTest(const unsigned int numberOfTrials,
                 const vector<double> & leftMatrices_serial,
                 const vector<double> & rightMatrices_serial,
                 const unsigned int matrixSize,
                 const unsigned int numberOfMatricesToMultiply,
                 Function function,
                 vector<double> * resultMatrices_serial_pointer,
                 double * elapsedTime) {

  *elapsedTime = std::numeric_limits<double>::max();

  for (unsigned int trialNumber = 0;
       trialNumber < numberOfTrials; ++trialNumber) {

    // Reset the cpu's cache
    Utilities::clearCpuCache();

    // Start measuring
    const high_resolution_clock::time_point tic = high_resolution_clock::now();

    // call the function
    function(leftMatrices_serial,
             rightMatrices_serial,
             matrixSize,
             numberOfMatricesToMultiply,
             resultMatrices_serial_pointer);

    // Stop measuring
    const high_resolution_clock::time_point toc = high_resolution_clock::now();
    const double thisTrialsElapsedTime =
      duration_cast<duration<double> >(toc - tic).count();
    // Take the minimum values from all trials
    *elapsedTime = std::min(*elapsedTime, thisTrialsElapsedTime);

  }

}

void
multiplyMatricesOnCpu_serial(const vector<double> & leftMatrices_serial,
                             const vector<double> & rightMatrices_serial,
                             const unsigned int matrixSize,
                             const unsigned int numberOfMatricesToMultiply,
                             vector<double> * resultMatrices_serial_pointer) {
  const unsigned int numberOfEntriesPerMatrix = matrixSize * matrixSize;
  vector<double> & resultMatrices_serial = *resultMatrices_serial_pointer;

  for (unsigned int matrixIndex = 0;
       matrixIndex < numberOfMatricesToMultiply; ++matrixIndex) {
    for (unsigned int row = 0; row < matrixSize; ++row) {
      const unsigned int leftBaseIndex =
        matrixIndex * numberOfEntriesPerMatrix + row * matrixSize;
      for (unsigned int col = 0; col < matrixSize; ++col) {
        const unsigned int rightBaseIndex =
          matrixIndex * numberOfEntriesPerMatrix + col;
        double sum = 0;
        for (unsigned int dummy = 0; dummy < matrixSize; ++dummy) {
          sum +=
            leftMatrices_serial[leftBaseIndex + dummy] *
            rightMatrices_serial[rightBaseIndex + dummy * matrixSize];
        }
        resultMatrices_serial[matrixIndex * numberOfEntriesPerMatrix +
                              row * matrixSize + col] = sum;
      }
    }
  }

}

void
multiplyMatricesOnCpu_omp(const vector<double> & leftMatrices_serial,
                          const vector<double> & rightMatrices_serial,
                          const unsigned int matrixSize,
                          const unsigned int numberOfMatricesToMultiply,
                          vector<double> * resultMatrices_serial_pointer) {
  const unsigned int numberOfEntriesPerMatrix = matrixSize * matrixSize;
  vector<double> & resultMatrices_serial = *resultMatrices_serial_pointer;

  // TODO: openmp-ize this

  for (unsigned int matrixIndex = 0;
       matrixIndex < numberOfMatricesToMultiply; ++matrixIndex) {
    for (unsigned int row = 0; row < matrixSize; ++row) {
      const unsigned int leftBaseIndex =
        matrixIndex * numberOfEntriesPerMatrix + row * matrixSize;
      for (unsigned int col = 0; col < matrixSize; ++col) {
        const unsigned int rightBaseIndex =
          matrixIndex * numberOfEntriesPerMatrix + col;
        double sum = 0;
        for (unsigned int dummy = 0; dummy < matrixSize; ++dummy) {
          sum +=
            leftMatrices_serial[leftBaseIndex + dummy] *
            rightMatrices_serial[rightBaseIndex + dummy * matrixSize];
        }
        resultMatrices_serial[matrixIndex * numberOfEntriesPerMatrix +
                              row * matrixSize + col] = sum;
      }
    }
  }

}

int main() {

  // ===============================================================
  // ********************** < Input> *******************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  const array<unsigned int, 2> rangeOfMatrixSizes = {{5, 1000}};
  const unsigned int numberOfDataPoints           = 10;
  const unsigned int numberOfThreadsPerBlock      = 1024;
  const unsigned int maxNumberOfBlocks            = 1e4;
  const unsigned int numberOfTrials               = 3;

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </Input> *******************************
  // ===============================================================

  std::vector<unsigned int> matrixSizes;
  for (unsigned int dataPointIndex = 0;
       dataPointIndex < numberOfDataPoints; ++dataPointIndex) {
    const size_t matrixSize =
      Utilities::interpolateNumberLinearlyOnLogScale(rangeOfMatrixSizes[0],
                                                     rangeOfMatrixSizes[1],
                                                     numberOfDataPoints,
                                                     dataPointIndex);
    if (matrixSizes.size() == 0 || matrixSize != matrixSizes.back()) {
      matrixSizes.push_back(matrixSize);
    }
  }

  const unsigned int cutoffMatrixSize = matrixSizes.back();
  const unsigned int cutoffNumberOfEntries = cutoffMatrixSize * cutoffMatrixSize;

  // create a random number generator
  std::mt19937 randomNumberEngine;
  std::uniform_real_distribution<double> randomNumberGenerator(0, 1);

  const string prefix = "data/ManyMatrixMultiplications_";
  const string suffix = "_shuffler";

  // make sure output directory exists
  std::ifstream test("data");
  if ((bool)test == false) {
    fprintf(stderr, "Error, cannot find data directory.  "
            "I don't like programs that make directories, so please make it "
            "yourself (\"mkdir data\")\n");
    exit(1);
  }

  char sprintfBuffer[500];
  sprintf(sprintfBuffer, "%stimes%s.csv", prefix.c_str(), suffix.c_str());
  FILE * file = fopen(sprintfBuffer, "w");
  fprintf(file, "matrixSize,numberOfMatricesToMultiply,serial,openmp,%s,%s\n",
          convertCudaManyMatrixMultiplicationStyleToString(NextThreadNextEntry_serialMatrices).c_str(),
          convertCudaManyMatrixMultiplicationStyleToString(NextThreadNextMatrix_deepEntries).c_str());
  for (const unsigned int matrixSize : matrixSizes) {

    const high_resolution_clock::time_point thisSizesTic =
      high_resolution_clock::now();

    // derive the number of entries
    const unsigned int numberOfEntriesPerMatrix = matrixSize * matrixSize;

    const unsigned int numberOfMatricesToMultiply =
      std::max(unsigned(3),
               unsigned(cutoffNumberOfEntries / numberOfEntriesPerMatrix));

    fprintf(file, "%10.4e, %10.4e",
            double(matrixSize),
            double(numberOfMatricesToMultiply));

    // create operands
    vector<double> leftMatrices_serial(numberOfMatricesToMultiply * numberOfEntriesPerMatrix);
    vector<double> rightMatrices_serial(numberOfMatricesToMultiply * numberOfEntriesPerMatrix);
    vector<double> resultMatrices_serial(numberOfMatricesToMultiply * numberOfEntriesPerMatrix);
    for (unsigned int matrixIndex = 0;
         matrixIndex < numberOfMatricesToMultiply; ++matrixIndex) {
      for (unsigned int row = 0; row < matrixSize; ++row) {
        for (unsigned int col = 0; col < matrixSize; ++col) {
          const double leftNumber =
            randomNumberGenerator(randomNumberEngine);
          leftMatrices_serial[matrixIndex * numberOfEntriesPerMatrix +
                              row * matrixSize + col] = leftNumber;
          const double rightNumber =
            randomNumberGenerator(randomNumberEngine);
          rightMatrices_serial[matrixIndex * numberOfEntriesPerMatrix +
                               row * matrixSize + col] = rightNumber;
        }
      }
    }

    double elapsedTime = 0;

    // ===============================================================
    // ******************* < do serial version> **********************
    // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    runCpuTimingTest(numberOfTrials,
                     leftMatrices_serial,
                     rightMatrices_serial,
                     matrixSize,
                     numberOfMatricesToMultiply,
                     multiplyMatricesOnCpu_serial,
                     &resultMatrices_serial,
                     &elapsedTime);
    fprintf(file, ", %10.4e", elapsedTime);

    const vector<double> correctResultMatrices = resultMatrices_serial;

    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // ******************* </do serial version> **********************
    // ===============================================================

    elapsedTime = std::numeric_limits<double>::quiet_NaN();
    std::fill(resultMatrices_serial.begin(),
              resultMatrices_serial.end(),
              std::numeric_limits<double>::quiet_NaN());

    // ===============================================================
    // *************** < do openmp > *********************************
    // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    runCpuTimingTest(numberOfTrials,
                     leftMatrices_serial,
                     rightMatrices_serial,
                     matrixSize,
                     numberOfMatricesToMultiply,
                     multiplyMatricesOnCpu_omp,
                     &resultMatrices_serial,
                     &elapsedTime);
    checkResult(correctResultMatrices, resultMatrices_serial,
                numberOfMatricesToMultiply, matrixSize,
                string("openmp"));
    fprintf(file, ", %10.4e", elapsedTime);

    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // *************** </do openmp > *********************************
    // ===============================================================

    elapsedTime = std::numeric_limits<double>::quiet_NaN();
    std::fill(resultMatrices_serial.begin(),
              resultMatrices_serial.end(),
              std::numeric_limits<double>::quiet_NaN());

    // ===============================================================
    // ******************* < do cuda serial matrices > ***************
    // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    // call cuda function
    runGpuTimingTest(numberOfTrials,
                     maxNumberOfBlocks,
                     numberOfThreadsPerBlock,
                     NextThreadNextEntry_serialMatrices,
                     numberOfMatricesToMultiply,
                     matrixSize,
                     &leftMatrices_serial[0],
                     &rightMatrices_serial[0],
                     &resultMatrices_serial[0],
                     &elapsedTime);

    checkResult(correctResultMatrices, resultMatrices_serial,
                numberOfMatricesToMultiply, matrixSize,
                convertCudaManyMatrixMultiplicationStyleToString(NextThreadNextEntry_serialMatrices).c_str());
    fprintf(file, ", %10.4e", elapsedTime);

    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // ******************* </do cuda serial matrices > ***************
    // ===============================================================

    elapsedTime = std::numeric_limits<double>::quiet_NaN();
    std::fill(resultMatrices_serial.begin(),
              resultMatrices_serial.end(),
              std::numeric_limits<double>::quiet_NaN());

    // ===============================================================
    // ******************* < do cuda deep entries> *******************
    // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    {

      vector<double> leftMatrices_deepEntries(leftMatrices_serial.size());
      vector<double> rightMatrices_deepEntries(rightMatrices_serial.size());
      vector<double> resultMatrices_deepEntries(resultMatrices_serial.size());

      for (unsigned int matrixIndex = 0;
           matrixIndex < numberOfMatricesToMultiply; ++matrixIndex) {
        for (unsigned int row = 0; row < matrixSize; ++row) {
          for (unsigned int col = 0; col < matrixSize; ++col) {
            const unsigned int indexWithinMatrix = row * matrixSize + col;
            leftMatrices_deepEntries[indexWithinMatrix * numberOfMatricesToMultiply +
                                     matrixIndex] =
              leftMatrices_serial[matrixIndex * numberOfEntriesPerMatrix +
                                  row * matrixSize + col];
            rightMatrices_deepEntries[indexWithinMatrix * numberOfMatricesToMultiply +
                                      matrixIndex] =
              rightMatrices_serial[matrixIndex * numberOfEntriesPerMatrix +
                                   row * matrixSize + col];
          }
        }
      }

      // call cuda function
      runGpuTimingTest(numberOfTrials,
                       maxNumberOfBlocks,
                       numberOfThreadsPerBlock,
                       NextThreadNextMatrix_deepEntries,
                       numberOfMatricesToMultiply,
                       matrixSize,
                       &leftMatrices_deepEntries[0],
                       &rightMatrices_deepEntries[0],
                       &resultMatrices_deepEntries[0],
                       &elapsedTime);

      // translate back to checkable structure
      for (unsigned int matrixIndex = 0;
           matrixIndex < numberOfMatricesToMultiply; ++matrixIndex) {
        for (unsigned int row = 0; row < matrixSize; ++row) {
          for (unsigned int col = 0; col < matrixSize; ++col) {
            const unsigned int indexWithinMatrix = row * matrixSize + col;
            resultMatrices_serial[matrixIndex * numberOfEntriesPerMatrix +
                                  row * matrixSize + col] =
              resultMatrices_deepEntries[indexWithinMatrix * numberOfMatricesToMultiply +
                                         matrixIndex];
          }
        }
      }

      checkResult(correctResultMatrices, resultMatrices_serial,
                  numberOfMatricesToMultiply, matrixSize,
                  convertCudaManyMatrixMultiplicationStyleToString(NextThreadNextMatrix_deepEntries).c_str());
      fprintf(file, ", %10.4e", elapsedTime);

    }

    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // ******************* </do cuda deep entries> *******************
    // ===============================================================

    fprintf(file, "\n");

    const high_resolution_clock::time_point thisSizesToc =
      high_resolution_clock::now();
    const double thisSizesElapsedTime =
      duration_cast<duration<double> >(thisSizesToc - thisSizesTic).count();
    printf("processed %8.2e matrices of size of %4u with %8.2e trials in "
           "%6.1f seconds\n", float(numberOfMatricesToMultiply),
           matrixSize, float(numberOfTrials), thisSizesElapsedTime);

  }

  fclose(file);

  return 0;
}
