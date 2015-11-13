// -*- C++ -*-
// Main4.cc
// cs181j hw9 Problem 4
// Calculating polynomials with cpu and gpu

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

// These utilities are used on many assignments
#include "../Utilities.h"

using std::string;
using std::vector;
using std::array;
using std::size_t;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

#include "Main4_cuda.cuh"

void
calculatePolynomials_cpu(const vector<float> & input,
                         const vector<float> & coefficients,
                         vector<float> * output) {
  const unsigned int inputSize = input.size();
  const unsigned int polynomialOrder = coefficients.size();
  vector<float> & outputReference = *output;

  // for each value
  for (unsigned int index = 0; index < inputSize; ++index) {
    const float x = input[index];
    float currentPower = 1;
    float y = 0;
    for (unsigned int powerIndex = 0;
         powerIndex < polynomialOrder; ++powerIndex) {
      y += coefficients[powerIndex] * currentPower;
      currentPower *= x;
    }
    outputReference[index] = y;
  }
}

void
runCpuTimingTest(const unsigned int numberOfTrials,
                 const vector<float> & input,
                 const vector<float> & coefficients,
                 vector<float> * output,
                 double * elapsedTime) {

  *elapsedTime = std::numeric_limits<double>::max();

  for (unsigned int trialNumber = 0;
       trialNumber < numberOfTrials; ++trialNumber) {

    // Start measuring
    const high_resolution_clock::time_point tic = high_resolution_clock::now();

    // Do the calculation
    calculatePolynomials_cpu(input, coefficients, output);

    // Stop measuring
    const high_resolution_clock::time_point toc = high_resolution_clock::now();
    const double thisTrialsElapsedTime =
      duration_cast<duration<double> >(toc - tic).count();
    // Take the minimum values from all trials
    *elapsedTime = std::min(*elapsedTime, thisTrialsElapsedTime);
  }

}


void
checkResult(const vector<float> & correctResult,
            const vector<float> & testResult,
            const float absoluteErrorTolerance,
            const string & testName) {
  char sprintfBuffer[500];
  const unsigned int size = correctResult.size();
  if (testResult.size() != size) {
    sprintf(sprintfBuffer, "Error found: "
            "testResult.size() = %zu, but correctResult.size() = %zu, "
            "test named "
            BOLD_ON FG_RED "%s" RESET "\n",
            testResult.size(), correctResult.size(),
            testName.c_str());
    throw std::runtime_error(sprintfBuffer);
  }
  for (unsigned int i = 0; i < size; ++i) {
    if (std::isfinite(testResult[i]) == false) {
      sprintf(sprintfBuffer, "Error found: "
              "testResult[%u] = %lf, "
              "test named "
              BOLD_ON FG_RED "%s" RESET "\n",
              i, testResult[i],
              testName.c_str());
      throw std::runtime_error(sprintfBuffer);
    }
    const double absoluteError = std::abs(testResult[i] - correctResult[i]);
    if (absoluteError > absoluteErrorTolerance) {
      sprintf(sprintfBuffer, "Error found: "
              "testResult[%u] = %lf, but correctResult[%u] = %lf, "
              "test named "
              BOLD_ON FG_RED "%s" RESET "\n",
              i, testResult[i],
              i, correctResult[i],
              testName.c_str());
      throw std::runtime_error(sprintfBuffer);
    }
  }
}

int main() {

  // ===============================================================
  // ********************** < Input> *******************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  const array<float, 2> inputSizeRange        = {{1e3, 1e6}};
  const unsigned int numberOfInputSizes       = 10;
  const array<float, 2> polynomialOrderRange  = {{1, 100}};
  const unsigned int numberOfPolynomialOrders = 10;
  const unsigned int numberOfThreadsPerBlock  = 256;
  const unsigned int maxNumberOfBlocks        = 1e4;
  const unsigned int numberOfTrials           = 5;

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </Input> *******************************
  // ===============================================================

  const double absoluteErrorTolerance = 1e-4;

  std::mt19937 randomNumberEngine;
  std::uniform_real_distribution<float> randomNumberGenerator(-1, 1);

  char sprintfBuffer[500];

  sprintf(sprintfBuffer, "data/Main4_");
  const string prefix = sprintfBuffer;
  const string suffix = "_shuffler";

  vector<vector<double> >
    inputSizeMatrixForPlotting(numberOfInputSizes,
                               vector<double>(numberOfPolynomialOrders));
  vector<vector<double> >
    polynomialOrderMatrixForPlotting(numberOfInputSizes,
                                     vector<double>(numberOfPolynomialOrders));
  vector<vector<double> >
    cpuTimes(numberOfInputSizes,
             vector<double>(numberOfPolynomialOrders));
  vector<vector<double> >
    gpuTimes(numberOfInputSizes,
             vector<double>(numberOfPolynomialOrders));

  // for each size
  for (unsigned int sizeIndex = 0; sizeIndex < numberOfInputSizes; ++sizeIndex) {
    const size_t inputSize =
      Utilities::interpolateNumberLinearlyOnLogScale(inputSizeRange[0],
                                                     inputSizeRange[1],
                                                     numberOfInputSizes,
                                                     sizeIndex);

    // generate the input
    vector<float> input(inputSize);
    for (unsigned int index = 0; index < inputSize; ++index) {
      input[index] = randomNumberGenerator(randomNumberEngine);
    }

    const high_resolution_clock::time_point thisSizesTic =
      high_resolution_clock::now();

    for (unsigned int orderIndex = 0;
         orderIndex < numberOfPolynomialOrders; ++orderIndex) {

      const size_t polynomialOrder =
        Utilities::interpolateNumberLinearlyOnLogScale(polynomialOrderRange[0],
                                                       polynomialOrderRange[1],
                                                       numberOfPolynomialOrders,
                                                       orderIndex);

      // generate the input
      vector<float> coefficients(polynomialOrder);
      for (unsigned int order = 0; order < polynomialOrder; ++order) {
        coefficients[order] = randomNumberGenerator(randomNumberEngine);
      }

      // ===============================================================
      // ******************* < do cpu version> *************************
      // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

      vector<float> cpuOutput(inputSize,
                              std::numeric_limits<float>::quiet_NaN());
      runCpuTimingTest(numberOfTrials,
                       input,
                       coefficients,
                       &cpuOutput,
                       &cpuTimes[sizeIndex][orderIndex]);

      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ******************* </do cpu version> *************************
      // ===============================================================




      // ===============================================================
      // ******************* < do gpu version> *************************
      // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

      vector<float> gpuOutput(inputSize,
                              std::numeric_limits<float>::quiet_NaN());
      try {
        // TODO: you'll probably have to change this function call
        runGpuTimingTest(numberOfTrials,
                         maxNumberOfBlocks,
                         numberOfThreadsPerBlock,
                         &gpuTimes[sizeIndex][orderIndex]);
        checkResult(cpuOutput,
                    gpuOutput,
                    absoluteErrorTolerance,
                    std::string("gpu"));
      } catch (const std::exception & e) {
        fprintf(stderr, "error caught attempting input size %zu "
                "and polynomial order %zu\n", inputSize, polynomialOrder);
        throw;
      }

      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ******************* </do gpu version> *************************
      // ===============================================================

      inputSizeMatrixForPlotting[sizeIndex][orderIndex] =
        inputSize;
      polynomialOrderMatrixForPlotting[sizeIndex][orderIndex] =
        polynomialOrder;
    }

    const high_resolution_clock::time_point thisSizesToc =
      high_resolution_clock::now();
    const float thisSizesElapsedTime =
      duration_cast<duration<float> >(thisSizesToc - thisSizesTic).count();
    printf("finished size %8.2e with %8.2e repeats in %6.2f seconds\n",
           float(inputSize), float(numberOfTrials),
           thisSizesElapsedTime);
  }

  Utilities::writeMatrixToFile(inputSizeMatrixForPlotting,
                               prefix + string("inputSizeMatrixForPlotting") + suffix);
  Utilities::writeMatrixToFile(polynomialOrderMatrixForPlotting,
                               prefix + string("polynomialOrderMatrixForPlotting") + suffix);
  Utilities::writeMatrixToFile(cpuTimes,
                               prefix + string("cpuTimes") + suffix);
  Utilities::writeMatrixToFile(gpuTimes,
                               prefix + string("gpuTimes") + suffix);


  // now that we've done the speedups for various sizes, choose a specific
  //  size and examine the speedups versus maxNumberOfBlocks and
  //  numberOfThreadsPerBlock
  {

    const unsigned int inputSize = 1e6;
    const unsigned int polynomialOrder = 1e2;
    const vector<unsigned int> numbersOfThreadsPerBlock =
      {{32, 64, 128, 256, 512, 1024}};
    const vector<unsigned int> maxNumbersOfBlocks =
      {{10, 20, 40, 60, 80, 100, 500, 1000, 10000}};

    // generate the input
    vector<float> input(inputSize);
    for (unsigned int index = 0; index < inputSize; ++index) {
      input[index] = randomNumberGenerator(randomNumberEngine);
    }

    // generate the coefficients
    vector<float> coefficients(polynomialOrder);
    for (unsigned int order = 0; order < polynomialOrder; ++order) {
      coefficients[order] = randomNumberGenerator(randomNumberEngine);
    }

    // ===============================================================
    // ******************* < do cpu version> *************************
    // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    vector<float> cpuOutput(inputSize,
                            std::numeric_limits<float>::quiet_NaN());
    double cpuElapsedTime;
    runCpuTimingTest(numberOfTrials,
                     input,
                     coefficients,
                     &cpuOutput,
                     &cpuElapsedTime);

    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // ******************* </do cpu version> *************************
    // ===============================================================


    // ===============================================================
    // ******************* < do gpu version> *************************
    // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    vector<vector<double> >
      gpuSpeedups(maxNumbersOfBlocks.size(),
                  vector<double>(numbersOfThreadsPerBlock.size()));
    vector<vector<double> >
      maxNumberOfBlocksMatrixForPlotting(maxNumbersOfBlocks.size(),
                                         vector<double>(numbersOfThreadsPerBlock.size()));
    vector<vector<double> >
      numberOfThreadsMatrixForPlotting(maxNumbersOfBlocks.size(),
                                       vector<double>(numbersOfThreadsPerBlock.size()));

    for (unsigned int blockIndex = 0;
         blockIndex < maxNumbersOfBlocks.size(); ++blockIndex) {

      const unsigned int maxNumberOfBlocks =
        maxNumbersOfBlocks[blockIndex];

      const high_resolution_clock::time_point thisSizesTic =
        high_resolution_clock::now();

      for (unsigned int threadIndex = 0;
           threadIndex < numbersOfThreadsPerBlock.size(); ++threadIndex) {

        const unsigned int numberOfThreadsPerBlock =
          numbersOfThreadsPerBlock[threadIndex];


        double gpuElapsedTime;
        vector<float> gpuOutput(inputSize,
                                std::numeric_limits<float>::quiet_NaN());
        try {
          // TODO: you'll probably have to change this function call
          runGpuTimingTest(numberOfTrials,
                           maxNumberOfBlocks,
                           numberOfThreadsPerBlock,
                           &gpuElapsedTime);
          checkResult(cpuOutput,
                      gpuOutput,
                      absoluteErrorTolerance,
                      std::string("gpu"));
        } catch (const std::exception & e) {
          fprintf(stderr, "error caught attempting a max number of blocks of %u "
                  "and %u threads per block\n", maxNumberOfBlocks,
                  numberOfThreadsPerBlock);
          throw;
        }
        gpuSpeedups[blockIndex][threadIndex] = cpuElapsedTime / gpuElapsedTime;

        maxNumberOfBlocksMatrixForPlotting[blockIndex][threadIndex] =
          maxNumberOfBlocks;
        numberOfThreadsMatrixForPlotting[blockIndex][threadIndex] =
          numberOfThreadsPerBlock;

      }

      const high_resolution_clock::time_point thisSizesToc =
        high_resolution_clock::now();
      const float thisSizesElapsedTime =
        duration_cast<duration<float> >(thisSizesToc - thisSizesTic).count();
      printf("finished %8.2e blocks with %8.2e repeats in %6.2f seconds\n",
             float(maxNumberOfBlocks), float(numberOfTrials),
             thisSizesElapsedTime);

    }

    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // ******************* </do gpu version> *************************
    // ===============================================================

    Utilities::writeMatrixToFile(gpuSpeedups,
                                 prefix + string("gpuSpeedupsVersusBlocksAndThreads") + suffix);
    Utilities::writeMatrixToFile(maxNumberOfBlocksMatrixForPlotting,
                                 prefix + string("maxNumberOfBlocks") + suffix);
    Utilities::writeMatrixToFile(numberOfThreadsMatrixForPlotting,
                                 prefix + string("numberOfThreadsPerBlock") + suffix);

  }

  return 0;
}
