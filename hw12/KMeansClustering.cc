// -*- C++ -*-
// KMeansClustering.cc
// cs181j hw12
// An exercise in doing distributed-memory parallel programming on k-means
//  clustering.

// magic header for all mpi stuff
#include <mpi.h>

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

// These utilities are used on many assignments
#include "../Utilities.h"

#include "KMeansClustering_functors.h"

using std::vector;
using std::array;
using std::string;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

void
checkResult(const vector<Point> & correctResult,
            const vector<Point> & testResult,
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
      magnitude(correctResult[i] - testResult[i]);
    if (absoluteError > absoluteErrorTolerance) {
      sprintf(sprintfBuffer, "wrong result for centroid number %zu in test result, "
              "it's (%e, %e, %e) but should be (%e, %e, %e), test named "
              BOLD_ON FG_RED "%s" RESET "\n", i,
              testResult[i][0], testResult[i][1], testResult[i][2],
              correctResult[i][0], correctResult[i][1], correctResult[i][2],
              testName.c_str());
      throw std::runtime_error(sprintfBuffer);
    }
  }
}

int main(int argc, char * argv[]) {

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

  // ===========================================================================
  // *************************** < Inputs> *************************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  // A lot of homeworks will run something over a range of sizes,
  //  which will then be plotted by some script.
  // This controls how many points are used.
  const array<double, 2> rangeOfNumberOfPoints = {{5, 1e5}};
  // This number controls how many data points are tested
  const unsigned int numberOfPointsDataPoints = 10;
  // This controls how many centroids are used.
  const vector<unsigned int> rangeOfNumberOfCentroids =
    {{10, 100, 1000}};
  // In real k-means calculations, the centroid updates would happen
  //  until some condition is satisfied.  In this, we'll just iterate
  //  a fixed number of times, so that all methods do the same amount
  //  of work.
  const unsigned int numberOfIterations = 5;

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // *************************** </Inputs> *************************************
  // ===========================================================================

  // On each test, we need to make sure we get the same result.  A test will
  //  fail if the difference between any entry in our result is more than
  //  absoluteErrorTolerance different than entries we got with another method.
  const double absoluteErrorTolerance = 1e-4;

  // Make a random number generator
  std::default_random_engine randomNumberGenerator;

  // For each resolution data point
  for (unsigned int numberOfPointsDataPointIndex = 0;
       numberOfPointsDataPointIndex < numberOfPointsDataPoints;
       ++numberOfPointsDataPointIndex) {
    // Calculate the number of points so that it's linear on a
    //  log scale.
    const size_t numberOfPoints =
      Utilities::interpolateNumberLinearlyOnLogScale(rangeOfNumberOfPoints[0],
                                                     rangeOfNumberOfPoints[1],
                                                     numberOfPointsDataPoints,
                                                     numberOfPointsDataPointIndex);

    for (const size_t numberOfCentroids : rangeOfNumberOfCentroids) {

      // Prepare real distributions for generating initial points
      const double numberOfCentroidsPerSide =
        std::ceil(std::pow(numberOfCentroids, 1./3.));
      std::uniform_real_distribution<float> uniformRealDistribution(0, 1.);
      const double normalMean = (1. / numberOfCentroidsPerSide) / 2.;
      const double normalStandardDeviation = normalMean / 3.;
      std::uniform_real_distribution<float> normalDistribution(normalMean,
                                                               normalStandardDeviation);

      // Prepare points
      vector<Point> points;
      points.reserve(numberOfPoints);
      const unsigned int numberOfPointsPerCentroid =
        numberOfPoints / numberOfCentroids;
      for (unsigned int centroidIndex = 0;
           centroidIndex < numberOfCentroids; ++centroidIndex) {
        const Point centroid =
          {{uniformRealDistribution(randomNumberGenerator),
            uniformRealDistribution(randomNumberGenerator),
            uniformRealDistribution(randomNumberGenerator)}};
        for (unsigned int pointNumber = 0;
             pointNumber < numberOfPointsPerCentroid; ++pointNumber) {
          const Point variation =
            {{normalDistribution(randomNumberGenerator),
              normalDistribution(randomNumberGenerator),
              normalDistribution(randomNumberGenerator)}};
          const Point p =
            {{centroid[0] + variation[0],
              centroid[1] + variation[1],
              centroid[2] + variation[2]}};
          points.push_back(p);
        }
      }
      // Throw in random points until it's full
      while (points.size() != numberOfPoints) {
        points.push_back((Point) {{
              uniformRealDistribution(randomNumberGenerator),
                uniformRealDistribution(randomNumberGenerator),
                uniformRealDistribution(randomNumberGenerator)}});
      }
      // Compute starting locations for the centroids
      vector<Point> startingCentroids;
      startingCentroids.resize(numberOfCentroids);
      for (unsigned int centroidIndex = 0;
           centroidIndex < numberOfCentroids; ++centroidIndex) {
        startingCentroids[centroidIndex][0] =
          uniformRealDistribution(randomNumberGenerator);
        startingCentroids[centroidIndex][1] =
          uniformRealDistribution(randomNumberGenerator);
        startingCentroids[centroidIndex][2] =
          uniformRealDistribution(randomNumberGenerator);
      }

      // do serial
      SerialClusterer serialClusterer(numberOfCentroids);
      vector<Point> serialFinalCentroids;
      serialFinalCentroids.resize(numberOfCentroids);
      serialClusterer.calculateClusterCentroids(numberOfIterations,
                                                points,
                                                startingCentroids,
                                                &serialFinalCentroids);

      // do mpi
      MPIClusterer mpiClusterer(numberOfCentroids,
                                rank,
                                numberOfProcesses);
      vector<Point> mpiFinalCentroids;
      mpiFinalCentroids.resize(numberOfCentroids);
      mpiClusterer.calculateClusterCentroids(numberOfIterations,
                                             points,
                                             startingCentroids,
                                             &mpiFinalCentroids);

      try {
        checkResult(serialFinalCentroids,
                    mpiFinalCentroids,
                    string("mpi"),
                    absoluteErrorTolerance);
      } catch (const std::exception & e) {
        fprintf(stderr, "invalid answer for %zu points and %zu centroids\n",
                numberOfPoints, numberOfCentroids);
        throw;
      }
      if (rank == 0) {
        printf("mpi got same answer for %6zu points and %4zu centroids\n",
               numberOfPoints, numberOfCentroids);
      }
    }
  }

  return 0;
}
