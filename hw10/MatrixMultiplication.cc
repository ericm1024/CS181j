// -*- C++ -*-
// MatrixMultiplication.cc
// cs181j hw10
// In this exercise, we experiment with matrix multiplication on the gpu

// These utilities are used on many assignments
#include "../Utilities.h"

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

using std::string;
using std::vector;
using std::array;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

#include "MatrixMultiplication_cuda.cuh"

void
checkResult(const vector<double> & correctResult,
            const vector<double> & testResult,
            const string & testName,
            const double absoluteErrorTolerance) {
  char sprintfBuffer[500];
  if (correctResult.size() != testResult.size()) {
    sprintf(sprintfBuffer, "test result has the wrong number of entries: %zu "
            "instead of %zu, test named "
            BOLD_ON FG_RED "%s" RESET "\n",
            testResult.size(), correctResult.size(),
            testName.c_str());
    throw std::runtime_error(sprintfBuffer);
  }
  for (size_t i = 0; i < correctResult.size(); ++i) {
    const double absoluteError =
      std::abs(correctResult[i] - testResult[i]);
    if (absoluteError > absoluteErrorTolerance) {
      sprintf(sprintfBuffer, "wrong result for matrix entry index %zu in "
              "test result, it's %e but should be %e, test named "
              BOLD_ON FG_RED "%s" RESET "\n", i,
              testResult[i], correctResult[i],
              testName.c_str());
      throw std::runtime_error(sprintfBuffer);
    }
  }
}

string
convertCudaMatrixMultiplicationStyleToString(const CudaMatrixMultiplicationStyle cudaMatrixMultiplicationStyle) {
  switch (cudaMatrixMultiplicationStyle) {
  case Naive_RowMajorTimesRowMajor:
    return string("Naive_RowMajorTimesRowMajor");
  case Naive_RowMajorTimesColMajor:
    return string("Naive_RowMajorTimesColMajor");
  case RowMajorStorage_TiledMultiplication_Global:
    return string("RowMajorStorage_TiledMultiplication_Global");
  default:
    fprintf(stderr, "invalid cuda matrix multiplication style\n");
    exit(1);
  };
}

template <class Function>
void
runCpuTimingTest(const unsigned int numberOfTrials,
                 const vector<double> & rowMajorLeftMatrix,
                 const vector<double> & rowMajorRightMatrix,
                 const unsigned int matrixSize,
                 Function function,
                 vector<double> * rowMajorResultMatrixPointer,
                 double * elapsedTime) {

  *elapsedTime = std::numeric_limits<double>::max();

  for (unsigned int trialNumber = 0;
       trialNumber < numberOfTrials; ++trialNumber) {

    // Reset the cpu's cache
    Utilities::clearCpuCache();

    // Start measuring
    const high_resolution_clock::time_point tic = high_resolution_clock::now();

    // call the function
    function(rowMajorLeftMatrix,
             rowMajorRightMatrix,
             matrixSize,
             rowMajorResultMatrixPointer);

    // Stop measuring
    const high_resolution_clock::time_point toc = high_resolution_clock::now();
    const double thisTrialsElapsedTime =
      duration_cast<duration<double> >(toc - tic).count();
    // Take the minimum values from all trials
    *elapsedTime = std::min(*elapsedTime, thisTrialsElapsedTime);

  }

}

void
multiplyRowMajorMatricesOnCpu(const vector<double> & rowMajorLeftMatrix,
                              const vector<double> & rowMajorRightMatrix,
                              const unsigned int matrixSize,
                              vector<double> * rowMajorResultMatrixPointer) {
  vector<double> & rowMajorResultMatrix = *rowMajorResultMatrixPointer;
  for (unsigned int row = 0; row < matrixSize; ++row) {
    for (unsigned int col = 0; col < matrixSize; ++col) {
      double sum = 0;
      for (unsigned int dummy = 0; dummy < matrixSize; ++dummy) {
        sum += rowMajorLeftMatrix[row * matrixSize + dummy] *
          rowMajorRightMatrix[dummy * matrixSize + col];
      }
      rowMajorResultMatrix[row * matrixSize + col] = sum;
    }
  }
}

int main() {

  // ===============================================================
  // ********************** < Input> *******************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  const array<unsigned int, 2> rangeOfMatrixSizes = {{32, 800}};
  const unsigned int numberOfDataPoints = 20;
  const unsigned int gpuTileSize = 32;
  const unsigned int numberOfThreadsPerBlock = gpuTileSize * gpuTileSize;
  const unsigned int maxNumberOfBlocks = 1e4;
  const unsigned int numberOfTrials = 5;

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </Input> *******************************
  // ===============================================================

  const double absoluteErrorTolerance = 1e-4;

  // generate matrix sizes
  std::vector<unsigned int> matrixSizes;
  for (unsigned int dataPointIndex = 0;
       dataPointIndex < numberOfDataPoints; ++dataPointIndex) {
    const size_t desiredMatrixSize =
      Utilities::interpolateNumberLinearlyOnLogScale(rangeOfMatrixSizes[0],
                                                     rangeOfMatrixSizes[1],
                                                     numberOfDataPoints,
                                                     dataPointIndex);
    const unsigned int matrixSize =
      unsigned(desiredMatrixSize / gpuTileSize) * gpuTileSize;
    if (matrixSizes.size() == 0 || matrixSize != matrixSizes.back()) {
      matrixSizes.push_back(matrixSize);
    }
  }

  // create a random number generator
  std::mt19937 randomNumberEngine;
  std::uniform_real_distribution<double> randomNumberGenerator(0, 1);

  const string prefix = "data/MatrixMultiplication_";
  const string suffix = "_shuffler";

  char sprintfBuffer[500];
  sprintf(sprintfBuffer, "%sresults%s.csv", prefix.c_str(), suffix.c_str());
  FILE * resultsFile = fopen(sprintfBuffer, "w");
  for (const unsigned int matrixSize : matrixSizes) {

    const high_resolution_clock::time_point thisSizesTic =
      high_resolution_clock::now();

    // derive the number of entries
    const unsigned int numberOfEntries = matrixSize * matrixSize;

    fprintf(resultsFile, "%10.4e", double(matrixSize));

    // create operands
    vector<double> rowMajorLeftMatrix(numberOfEntries);
    vector<double> rowMajorRightMatrix(numberOfEntries);
    vector<double> colMajorRightMatrix(numberOfEntries);
    vector<double> rowMajorResultMatrix(numberOfEntries);
    for (unsigned int row = 0; row < matrixSize; ++row) {
      for (unsigned int col = 0; col < matrixSize; ++col) {
        const double leftRandomNumber =
          randomNumberGenerator(randomNumberEngine);
        rowMajorLeftMatrix[row * matrixSize + col] = leftRandomNumber;
        const double rightRandomNumber =
          randomNumberGenerator(randomNumberEngine);
        colMajorRightMatrix[row + col * matrixSize] = rightRandomNumber;
        rowMajorRightMatrix[row * matrixSize + col] = rightRandomNumber;
      }
    }

    // ===============================================================
    // ********************** < do row*row> **************************
    // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    double rowRowElapsedTime = 0;
    runCpuTimingTest(numberOfTrials,
                     rowMajorLeftMatrix,
                     rowMajorRightMatrix,
                     matrixSize,
                     multiplyRowMajorMatricesOnCpu,
                     &rowMajorResultMatrix,
                     &rowRowElapsedTime);
    fprintf(resultsFile, ", %10.4e", rowRowElapsedTime);

    const vector<double> correctRowMajorResultMatrix = rowMajorResultMatrix;

    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // ********************** </do row*row> **************************
    // ===============================================================

    // ===============================================================
    // ********************** < do cuda flavors> *********************
    // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    const vector<CudaMatrixMultiplicationStyle> cudaRowMajorMatrixMultiplicationStyles =
      {{Naive_RowMajorTimesRowMajor,
        RowMajorStorage_TiledMultiplication_Global}};

    for (const CudaMatrixMultiplicationStyle cudaMatrixMultiplicationStyle :
           cudaRowMajorMatrixMultiplicationStyles) {

      // reset the result matrix
      std::fill(&rowMajorResultMatrix[0],
                &rowMajorResultMatrix[numberOfEntries], 0.);

      double elapsedTime = 0;
      runGpuTimingTest(numberOfTrials,
                       maxNumberOfBlocks,
                       numberOfThreadsPerBlock,
                       matrixSize,
                       gpuTileSize,
                       cudaMatrixMultiplicationStyle,
                       &rowMajorLeftMatrix[0],
                       &rowMajorRightMatrix[0],
                       &rowMajorResultMatrix[0],
                       &elapsedTime);
      // check the result
      checkResult(correctRowMajorResultMatrix,
                  rowMajorResultMatrix,
                  convertCudaMatrixMultiplicationStyleToString(cudaMatrixMultiplicationStyle),
                  absoluteErrorTolerance);
      fprintf(resultsFile, ", %10.4e", elapsedTime);
    }

    // run row * col
    {
      // reset the result matrix
      std::fill(&rowMajorResultMatrix[0],
                &rowMajorResultMatrix[numberOfEntries], 0.);

      double elapsedTime = 0;
      runGpuTimingTest(numberOfTrials,
                       maxNumberOfBlocks,
                       numberOfThreadsPerBlock,
                       matrixSize,
                       gpuTileSize,
                       Naive_RowMajorTimesColMajor,
                       &rowMajorLeftMatrix[0],
                       &colMajorRightMatrix[0],
                       &rowMajorResultMatrix[0],
                       &elapsedTime);
      // check the result
      checkResult(correctRowMajorResultMatrix,
                  rowMajorResultMatrix,
                  convertCudaMatrixMultiplicationStyleToString(Naive_RowMajorTimesColMajor),
                  absoluteErrorTolerance);
      fprintf(resultsFile, ", %10.4e", elapsedTime);
    }

    // run cublas
    {
      // reset the result matrix
      std::fill(&rowMajorResultMatrix[0],
                &rowMajorResultMatrix[numberOfEntries], 0.);

      double elapsedTime;
      multiplyRowMajorMatricesUsingCublas(numberOfTrials,
                                          matrixSize,
                                          &rowMajorLeftMatrix[0],
                                          &rowMajorRightMatrix[0],
                                          &rowMajorResultMatrix[0],
                                          &elapsedTime);
      // check the result
      checkResult(correctRowMajorResultMatrix,
                  rowMajorResultMatrix,
                  std::string("cublas"),
                  absoluteErrorTolerance);
      fprintf(resultsFile, ", %10.4e", elapsedTime);
    }

    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // ********************** </do cuda flavors> *********************
    // ===============================================================

    fprintf(resultsFile, "\n");

    const high_resolution_clock::time_point thisSizesToc =
      high_resolution_clock::now();
    const double thisSizesElapsedTime =
      duration_cast<duration<double> >(thisSizesToc - thisSizesTic).count();
    printf("processed a matrix size of %4u with %8.2e trials in %5.1f seconds\n",
           matrixSize, double(numberOfTrials), thisSizesElapsedTime);
  }

  fclose(resultsFile);

  return 0;
}
