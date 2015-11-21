// -*- C++ -*-
// ReductionRace.cc
// cs181j hw10
// In this exercise, use the gpu to do reductions

// These utilities are used on many assignments
#include "../Utilities.h"

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

using std::array;
using std::string;
using std::vector;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

#include "ReductionRace_cuda.cuh"

void
checkResult(const unsigned int correctResult,
            const unsigned int testResult,
            const string & testName) {
  char sprintfBuffer[500];
  if (testResult != correctResult) {
    sprintf(sprintfBuffer, "wrong result, %e instead of %e, test named "
            BOLD_ON FG_RED "%s" RESET "\n",
            double(testResult), double(correctResult),
            testName.c_str());
    throw std::runtime_error(sprintfBuffer);
  }
}

string
convertCudaReductionStyleToString(const CudaReductionStyle cudaStyle) {
  switch (cudaStyle) {
  case SerialBlockReduction:
    return string("SerialBlockReduction");
  case Stupendous:
    return string("Stupendous");
  default:
    throw std::runtime_error("invalid cuda reduction style");
  };
}

template <class Function>
void
runCpuTimingTest(const unsigned int numberOfTrials,
                 const vector<unsigned int> & input,
                 Function function,
                 unsigned int * output,
                 double * elapsedTime) {

  *elapsedTime = std::numeric_limits<double>::max();

  for (unsigned int trialNumber = 0;
       trialNumber < numberOfTrials; ++trialNumber) {

    // Reset the cpu's cache
    Utilities::clearCpuCache();

    // Start measuring
    const high_resolution_clock::time_point tic = high_resolution_clock::now();

    // call the function
    *output = function(input);

    // Stop measuring
    const high_resolution_clock::time_point toc = high_resolution_clock::now();
    const double thisTrialsElapsedTime =
      duration_cast<duration<double> >(toc - tic).count();
    // Take the minimum values from all trials
    *elapsedTime = std::min(*elapsedTime, thisTrialsElapsedTime);

  }

}

unsigned int
sumVector(const vector<unsigned int> & input) {
  const unsigned int inputSize = input.size();
  unsigned int output = 0;
  for (unsigned int index = 0; index < inputSize; ++index) {
    output += input[index];
  }
  return output;
}

int main() {

  // ===============================================================
  // ********************** < Input> *******************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  const array<unsigned int, 2> rangeOfVectorSizes = {{1000, 100000000}};
  const unsigned int numberOfDataPoints = 10;
  const vector<CudaReductionStyle> cudaStyles =
    {{SerialBlockReduction,
      Stupendous}};
  const unsigned int numberOfThreadsPerBlock = 1024;
  const vector<unsigned int> maxNumbersOfBlocks = {{80, 800, 8000}};
  const unsigned int numberOfTrials = 10;

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </Input> *******************************
  // ===============================================================

  char sprintfBuffer[500];

  const string prefix = "data/ReductionRace_";

  for (const unsigned int maxNumberOfBlocks : maxNumbersOfBlocks) {

    printf("running a maxNumberOfBlocks of %5u\n", maxNumberOfBlocks);

    sprintf(sprintfBuffer, "_shuffler_%05u", maxNumberOfBlocks);
    const string suffix = sprintfBuffer;

    sprintf(sprintfBuffer, "%stimes%s.csv", prefix.c_str(), suffix.c_str());
    FILE * file = fopen(sprintfBuffer, "w");
    fprintf(file, "inputSize,cpuSerial,thrust");
    for (const CudaReductionStyle cudaStyle : cudaStyles) {
      fprintf(file, ",%s", convertCudaReductionStyleToString(cudaStyle).c_str());
    }
    fprintf(file, "\n");

    for (unsigned int dataPointIndex = 0;
         dataPointIndex < numberOfDataPoints; ++dataPointIndex) {
      const size_t inputSize =
        Utilities::interpolateNumberLinearlyOnLogScale(rangeOfVectorSizes[0],
                                                       rangeOfVectorSizes[1],
                                                       numberOfDataPoints,
                                                       dataPointIndex);

      const high_resolution_clock::time_point thisSizesTic =
        high_resolution_clock::now();

      fprintf(file, "%10.4e", double(inputSize));

      // create operands
      vector<unsigned int> input(inputSize, 1);

      // ===============================================================
      // ******************* < do cpu version> *************************
      // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

      unsigned int cpuOutput;
      double cpuElapsedTime = 0;
      runCpuTimingTest(numberOfTrials,
                       input,
                       sumVector,
                       &cpuOutput,
                       &cpuElapsedTime);
      checkResult(inputSize,
                  cpuOutput,
                  std::string("cpu"));
      fprintf(file, ", %10.4e", cpuElapsedTime);

      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ******************* </do cpu version> *************************
      // ===============================================================

      // ===============================================================
      // ******************* < do thrust> ******************************
      // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

      double thrustElapsedTime;
      unsigned int thrustOutput;
      runThrustTest(numberOfTrials,
                    &input[0],
                    inputSize,
                    &thrustOutput,
                    &thrustElapsedTime);
      checkResult(inputSize,
                  thrustOutput,
                  std::string("thrust"));
      fprintf(file, ", %10.4e", thrustElapsedTime);

      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ******************* < do thrust> ******************************
      // ===============================================================

      // ===============================================================
      // ******************* < do gpu versions> ************************
      // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

      for (const CudaReductionStyle cudaStyle : cudaStyles) {

        double cudaElapsedTime;
        unsigned int cudaOutput;
        runGpuTimingTest(numberOfTrials,
                         maxNumberOfBlocks,
                         numberOfThreadsPerBlock,
                         cudaStyle,
                         &input[0],
                         inputSize,
                         &cudaOutput,
                         &cudaElapsedTime);
        checkResult(inputSize,
                    cudaOutput,
                    std::string("cuda ") + convertCudaReductionStyleToString(cudaStyle));
        fprintf(file, ", %10.4e", cudaElapsedTime);

      }

      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ******************* </do gpu versions> ************************
      // ===============================================================

      fprintf(file, "\n");

      const high_resolution_clock::time_point thisSizesToc =
        high_resolution_clock::now();
      const double thisSizesElapsedTime =
        duration_cast<duration<double> >(thisSizesToc - thisSizesTic).count();
      printf("finished inputSize of %8.2e in %6.2f seconds\n",
             double(inputSize), thisSizesElapsedTime);
    }

    fclose(file);

  }

  {
    const unsigned int inputSize = 1e7;
    const vector<unsigned int> numbersOfThreadsPerBlock =
      {{32, 64, 128, 256, 512, 1024}};
    const array<double, 2> rangeOfNumberOfBlocks = {{10, 10000}};
    const unsigned int numberOfBlocksDataPoints = 10;
    const string suffix = "_shuffler";

    // create operands
    vector<unsigned int> input(inputSize, 1);

    vector<vector<double> >
      cpuTimes(numberOfBlocksDataPoints,
               vector<double>(numbersOfThreadsPerBlock.size()));
    vector<vector<double> >
      gpuTimes(numberOfBlocksDataPoints,
               vector<double>(numbersOfThreadsPerBlock.size()));
    vector<vector<double> >
      maxNumberOfBlocksMatrixForPlotting(numberOfBlocksDataPoints,
                                         vector<double>(numbersOfThreadsPerBlock.size()));
    vector<vector<double> >
      numberOfThreadsMatrixForPlotting(numberOfBlocksDataPoints,
                                       vector<double>(numbersOfThreadsPerBlock.size()));

    // ===============================================================
    // ******************* < do cpu version> *************************
    // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    double cpuElapsedTime = 0;
    unsigned int cpuOutput;
    runCpuTimingTest(numberOfTrials,
                     input,
                     sumVector,
                     &cpuOutput,
                     &cpuElapsedTime);
    checkResult(inputSize,
                cpuOutput,
                std::string("cpu"));

    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // ******************* </do cpu version> *************************
    // ===============================================================

    for (unsigned int blockDataPointIndex = 0;
         blockDataPointIndex < numberOfBlocksDataPoints; ++blockDataPointIndex) {

      const size_t maxNumberOfBlocks =
        Utilities::interpolateNumberLinearlyOnLogScale(rangeOfNumberOfBlocks[0],
                                                       rangeOfNumberOfBlocks[1],
                                                       numberOfBlocksDataPoints,
                                                       blockDataPointIndex);

      const high_resolution_clock::time_point thisSizesTic =
        high_resolution_clock::now();

      for (unsigned int threadDataPointIndex = 0;
           threadDataPointIndex < numbersOfThreadsPerBlock.size();
           ++threadDataPointIndex) {

        const unsigned int numberOfThreadsPerBlock =
          numbersOfThreadsPerBlock[threadDataPointIndex];

        // ===============================================================
        // ******************* < do gpu version> *************************
        // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

        {
          unsigned int cudaOutput;
          runGpuTimingTest(numberOfTrials,
                           maxNumberOfBlocks,
                           numberOfThreadsPerBlock,
                           Stupendous,
                           &input[0],
                           inputSize,
                           &cudaOutput,
                           &gpuTimes[blockDataPointIndex][threadDataPointIndex]);
          checkResult(inputSize,
                      cudaOutput,
                      std::string("cuda ") + convertCudaReductionStyleToString(Stupendous));
        }

        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        // ******************* </do gpu version> *************************
        // ===============================================================

        cpuTimes[blockDataPointIndex][threadDataPointIndex] = cpuElapsedTime;
        maxNumberOfBlocksMatrixForPlotting[blockDataPointIndex][threadDataPointIndex] =
          maxNumberOfBlocks;
        numberOfThreadsMatrixForPlotting[blockDataPointIndex][threadDataPointIndex] =
          numberOfThreadsPerBlock;
      }

      const high_resolution_clock::time_point thisSizesToc =
        high_resolution_clock::now();
      const double thisSizesElapsedTime =
        duration_cast<duration<double> >(thisSizesToc - thisSizesTic).count();
      printf("finished %8.2e blocks with %8.2e trials in %6.2f seconds\n",
             double(maxNumberOfBlocks), double(numberOfTrials),
             thisSizesElapsedTime);
    }

    Utilities::writeMatrixToFile(cpuTimes,
                                 prefix + string("cpuTimes") + suffix);
    Utilities::writeMatrixToFile(gpuTimes,
                                 prefix + string("gpuTimes") + suffix);
    Utilities::writeMatrixToFile(maxNumberOfBlocksMatrixForPlotting,
                                 prefix + string("maxNumberOfBlocks") + suffix);
    Utilities::writeMatrixToFile(numberOfThreadsMatrixForPlotting,
                                 prefix + string("numberOfThreadsPerBlock") + suffix);

  }

  return 0;
}
