// -*- C++ -*-
// Mini2dMD_NeedToKnow.cc
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
           "\"mpirun -np X ./Mini2dMD_NeedToKnow\"\n");
  }
  const unsigned int numberOfProcessesPerSide = std::sqrt(numberOfProcesses);
  if (numberOfProcessesPerSide * numberOfProcessesPerSide !=
      numberOfProcesses) {
    fprintf(stderr, "invalid number of processes = %u, it must be a square "
            "such as 4, 9, 16, or 25\n", numberOfProcesses);
    exit(1);
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

  // ===========================================================================
  // *************************** < Parallelism> ********************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  // TODO: you probably want to change this.
  const BoundingBox thisRanksBoundingBox = simulationDomain;
  for (unsigned int processIndex = 0; processIndex < numberOfProcesses;
       ++processIndex) {
    if (processIndex == rank) {
      printf("%3u bounding box is (%7.2f, %7.2f) x (%7.2f, %7.2f)\n", rank,
             thisRanksBoundingBox._lower[0],
             thisRanksBoundingBox._lower[1],
             thisRanksBoundingBox._upper[0],
             thisRanksBoundingBox._upper[1]);
      fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // *************************** </Parallelism> ********************************
  // ===========================================================================

  unsigned int randomNumberSeed = 0;
  //const unsigned int randomNumberSeed = time(0);
  MPI_Bcast(&randomNumberSeed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  std::mt19937 randomNumberEngine(randomNumberSeed);

  // ===========================================================================
  // ************************* < Initial Condition> ****************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  vector<Point> globalPositions;
  vector<Vector> globalVelocities;
  Mini2dMD::generateInitialCondition(simulationDomain,
                                     globalNumberOfPoints,
                                     initialVelocityMaxMagnitude,
                                     &randomNumberEngine,
                                     &globalPositions,
                                     &globalVelocities);
  vector<Point> positions;
  vector<Vector> velocities;
  // TODO: only keep the stuff in our bounding box

  unsigned int numberOfAtomsOnThisRank = positions.size();
  unsigned int totalNumberOfOwnedAtomsAcrossTheSimulation;
  MPI_Allreduce(&numberOfAtomsOnThisRank,
                &totalNumberOfOwnedAtomsAcrossTheSimulation,
                1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
  printf("starting with %u total positions\n",
         totalNumberOfOwnedAtomsAcrossTheSimulation);
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ************************* </Initial Condition> ****************************
  // ===========================================================================

  const high_resolution_clock::time_point totalTic =
    high_resolution_clock::now();
  // now that we have initial positions, for each timestep
  vector<vector<unsigned int> > neighborhoods;
  vector<Vector> accelerations(positions.size());
  unsigned int currentFileIndex = 0;
  for (unsigned int timestepIndex = 0;
       timestepIndex < numberOfTimesteps; ++timestepIndex) {

    if (positions.size() != velocities.size() ||
        positions.size() != accelerations.size()) {
      throwException("inconsistent sizes of positions (%zu), velocities (%zu), "
                     "and accelerations (%zu) at beginning of timestep\n",
                     positions.size(), velocities.size(), accelerations.size());
    }

    // =========================================================================
    // ******************** < Transfer Shadow Particles> ***********************
    // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    vector<Point> shadowPositions;
    vector<Point> shadowVelocities;
    // TODO send some of our positions (and their velocities) to others.
    // receive shadow information from others, store in shadowPositions
    //  and shadowVelocities

    // you can do it!

    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // ******************** </Transfer Shadow Particles> ***********************
    // =========================================================================

    // =========================================================================
    // ********************** < Calculate New Positions> ***********************
    // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    // NOTE: you shouldn't have to change anything in this section

    // make structures of all of the points and velocities
    vector<Point> ownedAndShadowPositions = positions;
    ownedAndShadowPositions.reserve(positions.size() + shadowPositions.size());
    std::copy(shadowPositions.begin(), shadowPositions.end(),
              std::back_inserter(ownedAndShadowPositions));
    vector<Point> ownedAndShadowVelocities = velocities;
    ownedAndShadowVelocities.reserve(velocities.size() + shadowVelocities.size());
    std::copy(shadowVelocities.begin(), shadowVelocities.end(),
              std::back_inserter(ownedAndShadowVelocities));

    // find neighbors
    Mini2dMD::findNeighbors(ownedAndShadowPositions, cutoffRadius, &neighborhoods);

    const unsigned int thisRanksStartingIndex = 0;
    const unsigned int thisRanksEndingIndex = positions.size();

    // calculate forces on our positions
    try {
      Mini2dMD::calculateForcesAndNewVelocities(ownedAndShadowPositions,
                                                ownedAndShadowVelocities,
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

    // sanity checks
    Vector thisRanksTotalVelocity = {{0, 0}};
    for (unsigned int pointIndex = 0;
         pointIndex < positions.size(); ++pointIndex) {
      thisRanksTotalVelocity[0] += velocities[pointIndex][0];
      thisRanksTotalVelocity[1] += velocities[pointIndex][1];
    }
    Vector globalTotalVelocity = {{0, 0}};
    MPI_Allreduce(&thisRanksTotalVelocity[0], &globalTotalVelocity[0],
                  2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const double totalVelocityMagnitude =
      std::sqrt(globalTotalVelocity[0] * globalTotalVelocity[0] +
                globalTotalVelocity[1] * globalTotalVelocity[1]);

    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // ********************** </Calculate New Positions> ***********************
    // =========================================================================

    // =========================================================================
    // ******************** < Transfer Owned Particles> ************************
    // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    // now, some of our particles have drifted out of our bounding box and
    //  into the bounding box of some other process.
    // TODO: send those particles to their new owners.

    // you can do it!

    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // ******************** </Transfer Owned Particles> ************************
    // =========================================================================

    // =========================================================================
    // ******************** < Write Output File> *******************************
    // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    // if we're on a timestep to write output files, write an output file
    if (numberOfOutputFiles > 0 &&
        (timestepIndex % outputFileWriteTimestepInterval == 0)) {

      // first, write normal output file
      char sprintfBuffer[500];
      sprintf(sprintfBuffer, "data/Mini2dMD_NeedToKnow_%03u_%05u_%03u.csv",
              numberOfProcesses, currentFileIndex, rank);
      if (rank == 0) {
        printf("%7u writing file %s, total velocity is (%11.4e, %11.4e) "
               "(%8.2e)\n", timestepIndex, sprintfBuffer,
               globalTotalVelocity[0], globalTotalVelocity[1],
               totalVelocityMagnitude);
      }
      FILE* file = fopen(sprintfBuffer, "w");
      fprintf(file, "%11.4e,%11.4e,0\n",
              simulationDomain._lower[0], simulationDomain._upper[0]);
      fprintf(file, "%11.4e,%11.4e,0\n",
              simulationDomain._lower[1], simulationDomain._upper[1]);
      for (unsigned int pointIndex = 0;
           pointIndex < positions.size(); ++pointIndex) {
        // i use this color to display shadow particles or not, but you don't
        //  have to do that.  if you want to, 0 is not shadowed,
        //  1 was transferred, and 2 is shadowed
        const unsigned int color = 0;
        fprintf(file, "%11.4e,%11.4e,%u\n",
                positions[pointIndex][0], positions[pointIndex][1], color);
      }
      fclose(file);

      // now, write the special per-rank debug output file
      sprintf(sprintfBuffer, "data/Mini2dMD_NeedToKnow_Debug_%03u_%05u_%03u.csv",
              numberOfProcesses, currentFileIndex, rank);
      file = fopen(sprintfBuffer, "w");
      fprintf(file, "%11.4e,%11.4e,0\n",
              simulationDomain._lower[0], simulationDomain._upper[0]);
      fprintf(file, "%11.4e,%11.4e,0\n",
              simulationDomain._lower[1], simulationDomain._upper[1]);
      // if you want special colors, make the color 0 if the particle belongs
      //  to this process but is not shadowed to other processes,
      //  1 if the point is shadowed to other processes, and 2 if the point
      //  is shadowed from another process to this process (i.e. this process
      //  doesn't own it)
      for (unsigned int pointIndex = 0;
           pointIndex < positions.size(); ++pointIndex) {
        const unsigned int color = 0;
        fprintf(file, "%11.4e,%11.4e,%u\n",
                positions[pointIndex][0], positions[pointIndex][1], color);
      }
      fclose(file);

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
