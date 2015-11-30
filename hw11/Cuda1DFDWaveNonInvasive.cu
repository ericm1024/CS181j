// -*- C++ -*-
// Cuda1DFDWaveNonInvasive.cu
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
  const unsigned int numberOfIntervals = 1e5;
  const unsigned int numberOfOutputFiles = 100;
  const double fractionOfRealSimulationTimeToActuallyDoBecauseThisIsSoPainfullySlow = 0.10;
  const double simulationTime = 2.0;
  const float courant = 1.0;
  const float omega0 = 10;
  const float omega1 = 100;
  const float omega2 = 300;
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

  // make 3 copies of the domain for old, current, and new displacements
  float ** cpuData = new float*[3];
  // fill cpu copy with zeros
  for (unsigned int i = 0; i < 3; ++i) {
    // allocate room on the device
    cpuData[i] = new float[numberOfNodes];
    std::fill(&cpuData[i][0], &cpuData[i][1], 0.);
  }

  float timeSpentWritingOutputFiles = 0;

  unsigned int currentFileIndex = 0;
  const TimeUtility::PreCpp11TimePoint totalTic = TimeUtility::getCurrentTime();

  for (size_t timestepIndex = 0;
       timestepIndex < numberOfTimesteps * fractionOfRealSimulationTimeToActuallyDoBecauseThisIsSoPainfullySlow;
       ++timestepIndex) {
    if (timestepIndex % (numberOfTimesteps / 10) == 0) {
      printf("Processing timestep %8zu (%5.1f%%)\n",
             timestepIndex, 100 * timestepIndex / float(numberOfTimesteps));
    }

    // nickname displacements
    const float * oldDisplacements     = cpuData[(timestepIndex + 2) % 3];
    const float * currentDisplacements = cpuData[(timestepIndex + 3) % 3];
    float * newDisplacements           = cpuData[(timestepIndex + 4) % 3];

    float * oldDisplacements_gpu;
    float * currentDisplacements_gpu;
    float * newDisplacements_gpu;

    const float t = timestepIndex * dt;
    const float leftBoundaryConditionValue =
      (omega0 * t < 2 * M_PI) ?
      0.8 * sin(omega0*t) + 0.2 * sin(omega1*t) + 0.075 * sin(omega2*t) : 0;

    // TODO allocate room for displacements on GPU

    // TODO copy information to GPU

    // TODO solve for new displacements and apply boundary condition on gpu

    // TODO copy information from GPU to CPU

    // TODO free room on GPU

    // if we should write an output file
    if (numberOfOutputFiles > 0 &&
        timestepIndex % outputFileWriteTimestepInterval == 0) {

      const TimeUtility::PreCpp11TimePoint tic = TimeUtility::getCurrentTime();

      float l2Norm = 0;
      for (unsigned int nodeIndex = 0; nodeIndex < numberOfNodes; ++nodeIndex) {
        l2Norm += newDisplacements[nodeIndex] * newDisplacements[nodeIndex];
      }
      l2Norm = std::sqrt(l2Norm);
      printf("writing output file %3u, l2 norm is %9.2e\n", currentFileIndex,
             l2Norm);

      // write output file
      char sprintfBuffer[500];
      sprintf(sprintfBuffer, "data/Cuda1DFDWaveNonInvasive_%05u.csv",
              currentFileIndex);
      FILE* file = fopen(sprintfBuffer, "w");
      // we don't need to display all the points, 1000 is sufficient to
      //  keep down our hard disk usage
      const unsigned int nodeIncrement =
        std::max(unsigned(1), numberOfNodes/1000);
      for (unsigned int nodeIndex = 0; nodeIndex < numberOfNodes;
           nodeIndex+=nodeIncrement) {
        fprintf(file, "%e,%e\n", nodeIndex * dx, newDisplacements[nodeIndex]);
      }
      fclose(file);
      const TimeUtility::PreCpp11TimePoint toc = TimeUtility::getCurrentTime();
      timeSpentWritingOutputFiles += TimeUtility::getElapsedTime(tic, toc);
      ++currentFileIndex;
    }
  }

  // free memory on the device
  for (unsigned int index = 0; index < 3; ++index) {
    delete[] cpuData[index];
  }
  delete[] cpuData;

  const TimeUtility::PreCpp11TimePoint totalToc = TimeUtility::getCurrentTime();
  const float totalElapsedTime = TimeUtility::getElapsedTime(totalTic, totalToc);
  printf("Elapsed time: %6.2f (%6.3f (%5.1f%%) writing %3u output files)\n",
         totalElapsedTime,
         timeSpentWritingOutputFiles,
         100. * timeSpentWritingOutputFiles / totalElapsedTime,
         currentFileIndex);
  printf("You can now turn the output files into pictures by running "
         "\"python generate1DFDWaveEquationPlots_cuda.py NonInvasive\". "
         "It should produce jpg "
         "files in the figures directory.  You can then actually make a movie "
         "by running \"sh MakeMovie.sh NonInvasive\".\n");
  printf("\033[33m\033[1mNote \033[0m that the simulation only ran for %5.1f%% "
         "of the normal amount of time because it's so freaking slow.  "
         "The projected total time would normally be %lf seconds\n",
         100. * fractionOfRealSimulationTimeToActuallyDoBecauseThisIsSoPainfullySlow,
         totalElapsedTime / fractionOfRealSimulationTimeToActuallyDoBecauseThisIsSoPainfullySlow);

  return 0;
}
