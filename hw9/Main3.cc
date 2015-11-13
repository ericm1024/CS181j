// -*- C++ -*-
// Main3.cc
// cs181j hw9 Problem 1
// A simple example of calculating a weird function on cpu and gpu

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

#include "Main3_cuda.cuh"

int main(int argc, char * argv[]) {

  // ===============================================================
  // ********************** < Input> *******************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  const unsigned int inputSize = 1e4;
  unsigned int numberOfThreadsPerBlock = 1024;
  // ugly, yes.
  if (argc > 1) {
    numberOfThreadsPerBlock = atoi(argv[1]);
  }
  unsigned int maxNumberOfBlocks = 1e2;
  // real command-line parsing is worse, trust me.
  if (argc > 2) {
    maxNumberOfBlocks = unsigned(atof(argv[2]));
  }

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </Input> *******************************
  // ===============================================================

  std::mt19937 randomNumberEngine;
  std::uniform_real_distribution<float> randomNumberGenerator(-1, 1);

  vector<double> input(inputSize);
  for (unsigned int index = 0; index < inputSize; ++index) {
    input[index] = randomNumberGenerator(randomNumberEngine);
  }

  // calculate weird function on the cpu
  vector<double> cpuOutput(inputSize);
  for (unsigned int index = 0; index < inputSize; ++index) {
    const double x = input[index];
    cpuOutput[index] = sin(cos(exp(tan(x))));
  }

  // do the same calculation on the gpu
  vector<double> gpuOutput(inputSize,
                          std::numeric_limits<double>::quiet_NaN());
  // TODO: you'll probably need to change this function call
  calculateWeirdFunction_cuda(maxNumberOfBlocks,
                              numberOfThreadsPerBlock);
  // check our answer
  for (unsigned int index = 0; index < inputSize; ++index) {
    if (std::isfinite(gpuOutput[index]) == false ||
        std::abs(cpuOutput[index] - gpuOutput[index]) > 1e-4) {
      fprintf(stderr, "error: different value in cpuOutput[%u] = %e than "
              "gpuOutput[%u] = %e\n",
              index, cpuOutput[index], index, gpuOutput[index]);
      exit(1);
    }
  }
  printf("apparently you got the same results on the gpu!  good job!\n");

  return 0;
}
