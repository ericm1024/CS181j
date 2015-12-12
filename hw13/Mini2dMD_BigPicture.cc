// -*- C++ -*-
// Mini2dMD_BigPicture.cc
// cs181j hw13
// This is a simple 2d md simulation with a completely bogus material model
//  just to go through the practice of parallelizing a somewhat realistic
//  scenario.

// header for all mpi stuff
#include <mpi.h>

#include "Mini2dMD.h"

using std::string;
using std::vector;
using std::array;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

using Mini2dMD::BoundingBox;

int main(int argc, char* argv[]) {

  // Initialize MPI
  int mpiErrorCode = MPI_Init(&argc, &argv);
  if (mpiErrorCode != MPI_SUCCESS) {
    fprintf(stderr, "error in MPI_Init; aborting...\n");
    exit(1);
  }

  // Figure out what rank I am
  int temp;
  MPI_Comm_rank(MPI_COMM_WORLD, &temp);
  const unsigned int rank = temp;
  MPI_Comm_size(MPI_COMM_WORLD, &temp);
  const unsigned int numberOfProcesses = temp;
  if (numberOfProcesses == 1) {
    printf("Remember, to run this with mpi, you need to do "
           "\"mpirun -np X ./Main0\"\n");
  }

  // ===========================================================================
  // *************************** < Inputs> *************************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  const double domainHeight                = 100;
  const double aspectRatio                 = 1920. / 1080.;
  const unsigned int globalNumberOfPoints  = 500;
  const unsigned int numberOfOutputFiles   = 250;
  const double timestep                    = 1e-3;
  const double simulationTime              = 100;
  const double equilibriumDistance         = 1;
  const double forceConstant               = 50;
  const double stickingTimescale           = 1e-2;
  const double mass                        = 1;
  const double initialVelocityMaxMagnitude = 10;
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // *************************** </Inputs> *************************************
  // ===========================================================================

  // ===========================================================================
  // *************************** < Derived> ************************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  const double domainWidth                 = aspectRatio * domainHeight;
  const BoundingBox simulationDomain((Point) {{-domainWidth / 2., -domainHeight / 2.}},
                                     (Point) {{ domainWidth / 2.,  domainHeight / 2.}});
  const double numberOfTimesteps = simulationTime / timestep;
  const double cutoffRadius = 2.5 * equilibriumDistance;
  const unsigned int outputFileWriteTimestepInterval =
    (numberOfOutputFiles == 0) ? 1 :
    std::max(unsigned(1), unsigned(numberOfTimesteps / numberOfOutputFiles));
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // *************************** </Derived> ************************************
  // ===========================================================================

  unsigned int randomNumberSeed = 0;
  MPI_Bcast(&randomNumberSeed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  std::mt19937 randomNumberEngine(randomNumberSeed);
  std::uniform_real_distribution<double> uniformRealGenerator(0, 1);

  // ===========================================================================
  // ************************* < Initial Condition> ****************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  vector<Point> positions;
  vector<Vector> velocities;
  Mini2dMD::generateInitialCondition(simulationDomain,
                                     globalNumberOfPoints,
                                     initialVelocityMaxMagnitude,
                                     &randomNumberEngine,
                                     &positions,
                                     &velocities);
  printf("starting with %zu total positions\n", positions.size());
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ************************* </Initial Condition> ****************************
  // ===========================================================================

  const high_resolution_clock::time_point totalTic =
    high_resolution_clock::now();
  // now that we have initial positions, for each timestep
  vector<vector<unsigned int> > neighborhoods;
  vector<Vector> accelerations(positions.size());
  vector<Vector> newVelocities(positions.size());
  unsigned int currentFileIndex = 0;
  for (unsigned int timestepIndex = 0;
       timestepIndex < numberOfTimesteps; ++timestepIndex) {

    try {

      if (positions.size() != velocities.size() ||
          positions.size() != accelerations.size()) {
        throwException("inconsistent sizes of positions (%zu), velocities (%zu), "
                       "and accelerations (%zu) at beginning of timestep\n",
                       positions.size(), velocities.size(), accelerations.size());
      }

      // =========================================================================
      // ********************** < Calculate New Positions> ***********************
      // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

      // TODO: change these
      const unsigned int thisRanksStartingIndex = 0;
      const unsigned int thisRanksEndingIndex = positions.size();

      // find neighbors
      Mini2dMD::findNeighbors(positions, cutoffRadius, &neighborhoods);

      // calculate forces on our positions
      try {
        Mini2dMD::calculateForcesAndNewVelocities(positions,
                                                  velocities,
                                                  neighborhoods,
                                                  cutoffRadius,
                                                  equilibriumDistance,
                                                  forceConstant,
                                                  stickingTimescale,
                                                  mass,
                                                  timestep,
                                                  thisRanksStartingIndex,
                                                  thisRanksEndingIndex,
                                                  &velocities,
                                                  &accelerations);
      } catch (const std::exception & e) {
        fprintf(stderr, "something went wrong "
                "calculating forces and new velocities.\n");
        throw;
      }

      // integrate new positions and apply boundary conditions
      try {
        Mini2dMD::integrateVelocitiesAndReflectParticlesOnWalls(simulationDomain,
                                                                accelerations,
                                                                thisRanksStartingIndex,
                                                                thisRanksEndingIndex,
                                                                timestep,
                                                                &velocities,
                                                                &positions);
      } catch (const std::exception & e) {
        fprintf(stderr, "something went wrong "
                "integrating positions and applying boundary conditions.\n");
        throw;
      }

      // TODO: combine the results from all of the ranks

      // sanity checks
      Vector globalTotalVelocity = {{0, 0}};
      for (unsigned int pointIndex = 0;
           pointIndex < positions.size(); ++pointIndex) {
        globalTotalVelocity[0] += velocities[pointIndex][0];
        globalTotalVelocity[1] += velocities[pointIndex][1];
        if (std::isfinite(positions[pointIndex][0]) == false ||
            std::isfinite(positions[pointIndex][1]) == false) {
          fprintf(stderr, "timestep %u rank %3u, invalid position %u = "
                  "(%e, %e)\n", timestepIndex, rank, pointIndex,
                  positions[pointIndex][0], positions[pointIndex][1]);
          exit(1);
        }
      }
      const double totalVelocityMagnitude =
        std::sqrt(globalTotalVelocity[0] * globalTotalVelocity[0] +
                  globalTotalVelocity[1] * globalTotalVelocity[1]);

      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ********************** </Calculate New Positions> ***********************
      // =========================================================================


      // =========================================================================
      // ******************** < Write Output File> *******************************
      // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

      // if we're on a timestep to write output files, write an output file
      if (numberOfOutputFiles > 0 &&
          (timestepIndex % outputFileWriteTimestepInterval == 0)) {
        // write output file
        char sprintfBuffer[500];
        sprintf(sprintfBuffer, "data/Mini2dMD_BigPicture_%03u_%05u.csv",
                numberOfProcesses, currentFileIndex);
        if (rank == 0) {
          printf("%7u writing file %s, total velocity is (%11.4e, %11.4e) "
                 "(%8.2e)\n", timestepIndex, sprintfBuffer,
                 globalTotalVelocity[0], globalTotalVelocity[1],
                 totalVelocityMagnitude);
          FILE* file = fopen(sprintfBuffer, "w");
          fprintf(file, "%11.4e,%11.4e\n",
                  simulationDomain._lower[0], simulationDomain._upper[0]);
          fprintf(file, "%11.4e,%11.4e\n",
                  simulationDomain._lower[1], simulationDomain._upper[1]);
          for (unsigned int pointIndex = 0;
               pointIndex < positions.size(); ++pointIndex) {
            fprintf(file, "%11.4e,%11.4e\n",
                    positions[pointIndex][0], positions[pointIndex][1]);
          }
          fclose(file);
        }
        ++currentFileIndex;
      }

      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ******************** </Write Output File> *******************************
      // =========================================================================

      try {
        Mini2dMD::checkForInvalidValuesInState(positions,
                                               velocities,
                                               accelerations);
      } catch (const std::exception & e) {
        fprintf(stderr, "invalid value in state at end of timestep\n");
        throw;
      }

    } catch (const std::exception & e) {
      fprintf(stderr, "timestep %u rank %3u, something went wrong.\n",
              timestepIndex, rank);
      throw;
    }
  }

  const high_resolution_clock::time_point totalToc =
    high_resolution_clock::now();

  const double totalElapsedTime =
    duration_cast<duration<double> >(totalToc - totalTic).count();
  printf("simulation completed in %lf seconds\n", totalElapsedTime);

  // say goodbye
  MPI_Finalize();
  return 0;
}
