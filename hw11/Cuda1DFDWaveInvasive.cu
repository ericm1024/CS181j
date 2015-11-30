// -*- C++ -*-
// Cuda1DFDWaveInvasive.cu
// cs181j hw11
// A gpu-ized finite difference wave equation solver.

// c junk
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <ctime>

// c++ junk
#include <string>
#include <fstream>

// include cuda functions
#include <cuda_runtime.h>

// These utilities are used on many assignments
#include "../GpuUtilities.h"

__global__
void
calculateNewTimestep_kernel(const float * const __restrict__ oldDisplacements,
                            const float * const __restrict__ currentDisplacements,
                            // TODO?
                            float * const __restrict__ newDisplacements) {

  // TODO

}


int main(int argc, char* argv[]) {

  // make sure output directory exists
  std::ifstream test("data");
  if ((bool)test == false) {
    fprintf(stderr, "Error, cannot find data directory.  "
            "I don't like programs that make directories, so please make it "
            "yourself (\"mkdir data\")\n");
    exit(1);
  }

  // *************************** < Inputs> *************************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  const unsigned int numberOfIntervals   = 1e5;
  const unsigned int numberOfOutputFiles = 100;
  const double simulationTime            = 2.0;
  const float courant                    = 1.0;
  const float omega0                     = 10;
  const float omega1                     = 100;
  const float omega2                     = 300;
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // *************************** </Inputs> *************************************


  // *************************** < Derived> ************************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  const unsigned int numberOfNodes = numberOfIntervals + 1;
  const float courantSquared = courant * courant;
  const float c = 1;
  const float dx = 1./numberOfIntervals;
  const float dt = courant * dx / c;
  const unsigned int numberOfTimesteps = simulationTime / dt;
  const unsigned int outputFileWriteTimestepInterval =
    (numberOfOutputFiles == 0) ? 1 :
    std::max(unsigned(1), unsigned(numberOfTimesteps / numberOfOutputFiles));
  const unsigned int numberOfThreadsPerBlock = 1024;
  const unsigned int maxNumberOfBlocks = 1e4;
  const unsigned int numberOfBlocks =
    min(maxNumberOfBlocks,
        (unsigned int)ceil(numberOfNodes/float(numberOfThreadsPerBlock)));
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // *************************** </Derived> ************************************

  // TODO: data lives on the GPU, not the CPU.  only copy data to CPU in order
  //  to write files

  printf("You can now turn the output files into pictures by running "
         "\"python generate1DFDWaveEquationPlots_cuda.py Invasive\". "
         "It should produce jpg "
         "files in the figures directory.  You can then make a movie "
         "by running \"sh MakeMovie.sh Invasive\".\n");

  return 0;
}
