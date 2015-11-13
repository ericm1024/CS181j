// -*- C++ -*-
// Threaded1DFDWave.cc
// cs181j hw9 Problem 1
// A threaded finite difference wave equation solver.

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

// These utilities are used on many assignments
#include "../Utilities.h"

// this is nasty, but it gives us the number of cores on the machine
#include <unistd.h>

#include <deque>
#include <thread>

using std::string;
using std::vector;
using std::array;
using std::deque;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

int main(int argc, char* argv[]) {

  printf("Usage: ./Threaded1DFDWave numberOfThreads [numberOfIntervals]\n");

  // ===========================================================================
  // *************************** < Inputs> *************************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  unsigned int numberOfThreads = 1;
  unsigned int numberOfIntervals = 1e5;
  if (argc > 1) {
    numberOfThreads = atoi(argv[1]);
    if (argc > 2) {
      numberOfIntervals = atoi(argv[2]);
    } else {
      printf("You can specify the number of intervals with "
             "the second argument: "
             "./Threaded1DFDWave numberOfThreads [numberOfIntervals]\n");
    }
  } else {
    throw std::runtime_error("You need to specify the number of threads with "
                             "the first argument: "
                             "./Threaded1DFDWave numberOfThreads\n");
  }
  const unsigned int numberOfOutputFiles = 100;
  const double simulationTime = 2.0;
  const float courant = 1.0;
  const float omega0 = 10;
  const float omega1 = 100;
  const float omega2 = 300;
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // *************************** </Inputs> *************************************
  // ===========================================================================

  // ===========================================================================
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
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // *************************** </Derived> ************************************
  // ===========================================================================

  // Make sure that the data directory exists.
  Utilities::verifyThatDirectoryExists("data");

  char sprintfBuffer[500];

  float timeSpentWaitingOnFileOutput = 0;
  unsigned int currentFileIndex = 0;

  const high_resolution_clock::time_point totalTic =
    high_resolution_clock::now();

  // TODO: set the number of threads

  printf("Running with %8.2e intervals for %8.2e timesteps using %u threads\n",
         double(numberOfIntervals),
         double(numberOfTimesteps),
         numberOfThreads);

  // compute new displacements
  for (unsigned int timestepIndex = 0; timestepIndex < numberOfTimesteps;
       ++timestepIndex) {
    // TODO: maybe you only want one thread printing this.
    if (timestepIndex % (numberOfTimesteps / 10) == 0) {
      printf("Processing timestep %8u (%5.1f%%)\n",
             timestepIndex, 100 * timestepIndex / float(numberOfTimesteps));
    }
    // TODO: somehow, we need three timesteps worth of information
    const float * oldDisplacements;
    const float * currentDisplacements;
    float * newDisplacements;

    // TODO: compute newDisplacements[1] through
    //  newDisplacements[numberOfNodes-2] with the finite difference stencil

    // apply a wiggly boundary condition on the left side
    const float t = timestepIndex * dt;
    if (omega0 * t < 2 * M_PI) {
      // TODO: make this boundary condition something fun
      newDisplacements[0] =
        0.8 * sin(omega0 * t) + 0.2 * sin(omega1 * t) + 0.075 * sin(omega2 * t);
    } else {
      newDisplacements[0] = 0;
    }
    // right side clamped
    newDisplacements[numberOfNodes - 1] = 0;

    // enable this is you're having troubles with things exploding
#if 0
    // check health of the new displacements
    for (unsigned int nodeIndex = 0; nodeIndex < numberOfNodes; ++nodeIndex) {
      if (std::isfinite(newDisplacements[nodeIndex]) == false ||
          std::abs(newDisplacements[nodeIndex]) > 2) {
        printf("Error: bad displacement on timestep %u, node index %u: "
               "%10.4lf\n", timestepIndex, nodeIndex,
               newDisplacements[nodeIndex]);
      }
    }
#endif

    if (numberOfOutputFiles > 0 &&
        timestepIndex % outputFileWriteTimestepInterval == 0) {
      const high_resolution_clock::time_point tic =
        high_resolution_clock::now();
      // write output file
      sprintf(sprintfBuffer, "data/Threaded1DFDWave_%06u_%02u_%05u.csv",
              numberOfIntervals, numberOfThreads, currentFileIndex);
      FILE* file = fopen(sprintfBuffer, "w");
      // we don't need to display all the points, 1000 is sufficient to
      //  keep down our hard disk usage
      const unsigned int nodeIncrement =
        std::max(unsigned(1), numberOfNodes/1000);
      for (unsigned int nodeIndex = 0; nodeIndex < numberOfNodes;
           nodeIndex += nodeIncrement) {
        fprintf(file, "%e,%e\n", nodeIndex * dx, newDisplacements[nodeIndex]);
      }
      fclose(file);
      const high_resolution_clock::time_point toc =
        high_resolution_clock::now();
      const double thisFilesWritingTime =
        duration_cast<duration<double> >(toc - tic).count();
      timeSpentWaitingOnFileOutput += thisFilesWritingTime;
      ++currentFileIndex;
    }
  }

  const high_resolution_clock::time_point totalToc =
    high_resolution_clock::now();
  const double totalElapsedTime =
    duration_cast<duration<double> >(totalToc - totalTic).count();
  printf("Elapsed time: %7.3f (%6.3f (%5.1f%%) writing %3u output files\n",
         totalElapsedTime,
         timeSpentWaitingOnFileOutput,
         100. * timeSpentWaitingOnFileOutput / totalElapsedTime,
         currentFileIndex);
  printf("You can now turn the output files into pictures by running "
         "\"python generate1DFDWaveEquationPlots_threaded.py %u %u\". It should "
         "produce jpg files in the figures directory.  You can then make a "
         "movie by running \"sh MakeMovie.sh %u %u\".\n", numberOfThreads,
         numberOfIntervals, numberOfThreads, numberOfIntervals);

  return 0;
}
