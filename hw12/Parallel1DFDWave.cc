// -*- C++ -*-
// Parallel1DFDWave.cc
// cs181j hw12
// A distributed-memory parallel finite difference wave equation solver.

// magic header for all mpi stuff
#include <mpi.h>

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

int main(int argc, char* argv[]) {

  // make sure output directory exists
  Utilities::verifyThatDirectoryExists(std::string("data"));

  // *************************** < Inputs> *************************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  const unsigned int globalNumberOfIntervals = 1e5;
  const unsigned int numberOfOutputFiles     = 100;
  const double simulationTime                = 2.0;
  const float courant                        = 1.0;
  const float omega0                         = 10;
  const float omega1                         = 100;
  const float omega2                         = 300;
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // *************************** </Inputs> *************************************

  // *************************** < Derived> ************************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  const unsigned int globalNumberOfNodes = globalNumberOfIntervals + 1;
  const float courantSquared = courant * courant;
  const float c = 1;
  const float dx = 1./globalNumberOfIntervals;
  const float dt = courant * dx / c;
  const unsigned int numberOfTimesteps = simulationTime / dt;
  const unsigned int outputFileWriteTimestepInterval =
    (numberOfOutputFiles == 0) ? 1 :
    std::max(unsigned(1), unsigned(numberOfTimesteps / numberOfOutputFiles));
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // *************************** </Derived> ************************************

  // Initialize MPI
  int mpiErrorCode = MPI_Init(&argc, &argv);
  if (mpiErrorCode != MPI_SUCCESS) {
    printf("error in MPI_Init; aborting...\n");
    exit(1);
  }

  // Figure out what rank I am
  int temp;
  MPI_Comm_rank(MPI_COMM_WORLD, &temp);
  const unsigned int rank = temp;
  MPI_Comm_size(MPI_COMM_WORLD, &temp);
  const unsigned int numberOfProcesses = temp;

  // *************************** < Parallelism> ********************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  const double domainPerProcess = 1. / numberOfProcesses;
  // TODO: fix this
  const unsigned int numberOfNodesOnThisRank = 0;
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // *************************** </Parallelism> ********************************

  // make 3 copies of the domain for old, current, and new displacements
  float ** data = new float*[3];
  // fill cpu copy with zeros
  for (unsigned int i = 0; i < 3; ++i) {
    // allocate room on the device
    data[i] = new float[numberOfNodesOnThisRank];
    std::fill(&data[i][0], &data[i][1], 0.);
  }

  float timeSpentWritingOutputFiles = 0;

  unsigned int currentFileIndex = 0;
  const high_resolution_clock::time_point totalTic =
    high_resolution_clock::now();

  for (size_t timestepIndex = 0; timestepIndex < numberOfTimesteps;
       ++timestepIndex) {
    if (rank == 0 && timestepIndex % (numberOfTimesteps / 10) == 0) {
      printf("Processing timestep %8zu (%5.1f%%)\n",
             timestepIndex, 100 * timestepIndex / float(numberOfTimesteps));
    }

    // nickname displacements
    const float * oldDisplacements     = data[(timestepIndex + 2) % 3];
    const float * currentDisplacements = data[(timestepIndex + 3) % 3];
    float * newDisplacements           = data[(timestepIndex + 4) % 3];

    for (unsigned int nodeIndex = 1; nodeIndex < numberOfNodesOnThisRank - 1;
         ++nodeIndex) {
      const float spaceDerivative =
        currentDisplacements[nodeIndex - 1] -
        2 * currentDisplacements[nodeIndex] +
        currentDisplacements[nodeIndex + 1];
      newDisplacements[nodeIndex] =
        2 * currentDisplacements[nodeIndex] - oldDisplacements[nodeIndex] +
        courantSquared * spaceDerivative;
    }

    // TODO: apply a wiggly boundary condition on the left side
    const float t = timestepIndex * dt;
    // TODO: be careful, not all ranks should do this.
    if (omega0 * t < 2 * M_PI) {
      newDisplacements[0] =
        0.8 * sin(omega0 * t) + 0.2 * sin(omega1 * t) + 0.075 * sin(omega2 * t);
    } else {
      newDisplacements[0] = 0;
    }
    // TODO: right side clamped

    // TODO: communicate

    // enable this is you're having troubles with instabilities
#if 0
    // check health of the new displacements
    for (unsigned int nodeIndex = 0;
         nodeIndex < numberOfNodesOnThisRank; ++nodeIndex) {
      if (std::isfinite(newDisplacements[nodeIndex]) == false ||
          std::abs(newDisplacements[nodeIndex]) > 2) {
        printf("Error: bad displacement on timestep %u, node index %u: "
               "%10.4lf\n", timestepIndex, nodeIndex,
               newDisplacements[nodeIndex]);
      }
    }
#endif

    // if we should write an output file
    if (numberOfOutputFiles > 0 &&
        timestepIndex % outputFileWriteTimestepInterval == 0) {

      const high_resolution_clock::time_point tic =
        high_resolution_clock::now();

      // TODO: do you want to output the l2 norm for checking?
      // if so, you'll need to communicate here.

      // write output file
      char sprintfBuffer[500];
      sprintf(sprintfBuffer, "data/Parallel1DFDWave_%03u_%05u_%03u.csv",
              numberOfProcesses, currentFileIndex, rank);
      FILE* file = fopen(sprintfBuffer, "w");
      // we don't need to display all the points, 1000 is sufficient to
      //  see and it keeps down our hard disk usage
      const unsigned int nodeIncrement =
        std::max(unsigned(1), numberOfNodesOnThisRank/1000);
      for (unsigned int nodeIndex = 0; nodeIndex < numberOfNodesOnThisRank;
           nodeIndex+=nodeIncrement) {
        fprintf(file, "%e,%e\n", rank * domainPerProcess + nodeIndex * dx,
                newDisplacements[nodeIndex]);
      }
      fclose(file);
      const high_resolution_clock::time_point toc =
        high_resolution_clock::now();
      const double thisFilesWritingTime =
        duration_cast<duration<double> >(toc - tic).count();
      timeSpentWritingOutputFiles += thisFilesWritingTime;
      ++currentFileIndex;
    }
  }

  // free memory
  for (unsigned int index = 0; index < 3; ++index) {
    delete[] data[index];
  }
  delete[] data;

  const high_resolution_clock::time_point totalToc =
    high_resolution_clock::now();

  const double totalElapsedTime =
    duration_cast<duration<double> >(totalToc - totalTic).count();
  for (unsigned int processIndex = 0; processIndex < numberOfProcesses;
       ++processIndex) {
    if (processIndex == rank) {
      printf("Rank %2u elapsed time: %6.2f (%6.3f (%5.1f%%) writing %3u "
             "output files)\n",
             rank,
             totalElapsedTime,
             timeSpentWritingOutputFiles,
             100. * timeSpentWritingOutputFiles / totalElapsedTime,
             currentFileIndex);
      fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    printf("You can now turn the output files into pictures by running "
           "\"python generate1DFDWaveEquationPlots_mpi.py %u\". "
           "It should produce jpg files in the figures directory.  "
           "You can then make a movie "
           "by running \"sh MakeMovie.sh %u\".\n", numberOfProcesses,
           numberOfProcesses);
  }

  return 0;
}
