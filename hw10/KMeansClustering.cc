// -*- C++ -*-
// KMeansClustering.cc
// cs181j 2015 hw7
// An exercise in threading, accelerating the computation of k-means
//  clustering.

// These utilities are used on many assignments
#include "../Utilities.h"

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

#include "KMeansClustering_functors.h"

#include "KMeansClustering_cuda.cuh"

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
            const float absoluteErrorTolerance) {
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
    const float absoluteError =
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

template <class Clusterer>
void
runCpuTimingTest(const unsigned int numberOfTrials,
                 const unsigned int numberOfIterations,
                 const vector<Point> & startingCentroids,
                 Clusterer * clusterer,
                 vector<Point> * finalCentroids,
                 float * elapsedTime) {

  *elapsedTime = std::numeric_limits<float>::max();

  for (unsigned int trialNumber = 0;
       trialNumber < numberOfTrials; ++trialNumber) {

    // Reset intermediate values, i.e., nextCentroids and nextCentroidCounts
    clusterer->resetIntermediateValuesForNextCalculation(startingCentroids);

    // Reset the cpu's cache
    Utilities::clearCpuCache();

    // Start measuring
    const high_resolution_clock::time_point tic = high_resolution_clock::now();

    // Do the clustering
    clusterer->calculateClusterCentroids(numberOfIterations);

    // Stop measuring
    const high_resolution_clock::time_point toc = high_resolution_clock::now();
    const float thisTrialsElapsedTime =
      duration_cast<duration<float> >(toc - tic).count();
    // Take the minimum values from all trials
    *elapsedTime = std::min(*elapsedTime, thisTrialsElapsedTime);

    // Ship results back to Cpu for checking
    clusterer->moveResultsToCpu(finalCentroids);
  }

}

template <class RandomNumberGenerator>
void
generatePointsAndStartingCentroids(const unsigned int numberOfPoints,
                                   const unsigned int numberOfCentroids,
                                   RandomNumberGenerator * randomNumberGeneratorPointer,
                                   vector<Point> * points,
                                   vector<Point> * startingCentroids) {

  RandomNumberGenerator & randomNumberGenerator = *randomNumberGeneratorPointer;

  // Prepare real distributions for generating initial points
  const float numberOfCentroidsPerSide =
    std::ceil(std::pow(numberOfCentroids, 1./3.));
  std::uniform_real_distribution<float> uniformRealDistribution(0, 1.);
  const float normalMean = (1. / numberOfCentroidsPerSide) / 2.;
  const float normalStandardDeviation = normalMean / 3.;
  std::uniform_real_distribution<float> normalDistribution(normalMean,
                                                           normalStandardDeviation);

  // Prepare points
  points->reserve(numberOfPoints);
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
      points->push_back(p);
    }
  }
  // Throw in random points until it's full
  while (points->size() != numberOfPoints) {
    points->push_back((Point) {{
          uniformRealDistribution(randomNumberGenerator),
            uniformRealDistribution(randomNumberGenerator),
            uniformRealDistribution(randomNumberGenerator)}});
  }
  // Compute starting locations for the centroids
  startingCentroids->resize(numberOfCentroids);
  for (unsigned int centroidIndex = 0;
       centroidIndex < numberOfCentroids; ++centroidIndex) {
    startingCentroids->at(centroidIndex)[0] =
      uniformRealDistribution(randomNumberGenerator);
    startingCentroids->at(centroidIndex)[1] =
      uniformRealDistribution(randomNumberGenerator);
    startingCentroids->at(centroidIndex)[2] =
      uniformRealDistribution(randomNumberGenerator);
  }
}

int main() {

  // ===========================================================================
  // *************************** < Inputs> *************************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  // A lot of homeworks will run something over a range of sizes,
  //  which will then be plotted by some script.
  // This controls how many points are used.
  const array<float, 2> rangeOfNumberOfPoints = {{1e2, 1e6}};
  // This number controls how many data points are made and plotted.
  const unsigned int numberOfPointsDataPoints = 9;
  // This controls how many centroids are used.
  const array<float, 2> rangeOfNumberOfCentroids = {{5, 1e2}};
  // This number controls how many data points are made and plotted.
  const unsigned int numberOfCentroidsDataPoints = 6;
  // In real k-means calculations, the centroid updates would happen
  //  until some condition is satisfied.  In this, we'll just iterate
  //  a fixed number of times, so that all methods do the same amount
  //  of work.
  const unsigned int numberOfIterations = 2;
  // This is the standard number of times the calculation is repeated.
  const unsigned int numberOfTrials = 5;
  const unsigned int numberOfThreadsPerBlock = 256;
  const unsigned int maxNumberOfBlocks = 1e4;

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // *************************** </Inputs> *************************************
  // ===========================================================================

  const string prefix = "data/KMeansClustering_";
  const string suffix = "_shuffler";

  // On each test, we need to make sure we get the same result.  A test will
  //  fail if the difference between any entry in our result is more than
  //  absoluteErrorTolerance different than entries we got with another method.
  const float absoluteErrorTolerance = 1e-1;

  // Make sure that the data directory exists.
  Utilities::verifyThatDirectoryExists("data");

  // Make a random number generator
  std::default_random_engine randomNumberGenerator;

  // Prepare output matrices
  vector<vector<double> >
    numberOfPointsMatrixForPlotting(numberOfPointsDataPoints,
                                    vector<double>(numberOfCentroidsDataPoints, 0));
  vector<vector<double> >
    numberOfCentroidsMatrixForPlotting(numberOfPointsDataPoints,
                                       vector<double>(numberOfCentroidsDataPoints, 0));
  vector<vector<double> >
    cpuTimes(numberOfPointsDataPoints,
             vector<double>(numberOfCentroidsDataPoints, 0));
  vector<vector<double> >
    gpu_SoA_times(numberOfPointsDataPoints,
                  vector<double>(numberOfCentroidsDataPoints, 0));
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

    const high_resolution_clock::time_point thisSizesTic =
      high_resolution_clock::now();

    // for each number of centroids
    for (unsigned int numberOfCentroidsDataPointIndex = 0;
         numberOfCentroidsDataPointIndex < numberOfCentroidsDataPoints;
         ++numberOfCentroidsDataPointIndex) {
      // Calculate the number of points so that it's linear on a
      //  log scale.
      const size_t numberOfCentroids =
        Utilities::interpolateNumberLinearlyOnLinearScale(rangeOfNumberOfCentroids[0],
                                                          rangeOfNumberOfCentroids[1],
                                                          numberOfCentroidsDataPoints,
                                                          numberOfCentroidsDataPointIndex);

      vector<Point> points;
      vector<Point> startingCentroids;
      generatePointsAndStartingCentroids(numberOfPoints,
                                         numberOfCentroids,
                                         &randomNumberGenerator,
                                         &points,
                                         &startingCentroids);

      SerialClusterer serialClusterer(points, numberOfCentroids);

      vector<Point> serialFinalCentroids;
      serialFinalCentroids.resize(numberOfCentroids);
      float serialElapsedTime;
      runCpuTimingTest(numberOfTrials,
                       numberOfIterations,
                       startingCentroids,
                       &serialClusterer,
                       &serialFinalCentroids,
                       &serialElapsedTime);


      float gpu_SoA_elapsedTime;
      vector<Point> gpu_SoA_finalCentroids(numberOfCentroids);
      runGpuTimingTest(numberOfTrials,
                       maxNumberOfBlocks,
                       numberOfThreadsPerBlock,
                       &points[0][0],
                       numberOfPoints,
                       &startingCentroids[0][0],
                       numberOfCentroids,
                       numberOfIterations,
                       &gpu_SoA_finalCentroids[0][0],
                       &gpu_SoA_elapsedTime);
      checkResult(serialFinalCentroids,
                  gpu_SoA_finalCentroids,
                  string("gpu SoA"),
                  absoluteErrorTolerance);

      numberOfPointsMatrixForPlotting[numberOfPointsDataPointIndex][numberOfCentroidsDataPointIndex] =
        numberOfPoints;
      numberOfCentroidsMatrixForPlotting[numberOfPointsDataPointIndex][numberOfCentroidsDataPointIndex] =
        numberOfCentroids;
      cpuTimes[numberOfPointsDataPointIndex][numberOfCentroidsDataPointIndex] =
        serialElapsedTime;
      gpu_SoA_times[numberOfPointsDataPointIndex][numberOfCentroidsDataPointIndex] =
        gpu_SoA_elapsedTime;

    }
    const high_resolution_clock::time_point thisSizesToc =
      high_resolution_clock::now();
    const float thisSizesElapsedTime =
      duration_cast<duration<float> >(thisSizesToc -
                                      thisSizesTic).count();
    printf("processing %8.2e points took %7.2f seconds\n",
           float(numberOfPoints),
           thisSizesElapsedTime);
  }

  Utilities::writeMatrixToFile(numberOfPointsMatrixForPlotting,
                               prefix + string("numberOfPoints") + suffix);
  Utilities::writeMatrixToFile(numberOfCentroidsMatrixForPlotting,
                               prefix + string("numberOfCentroids") + suffix);
  Utilities::writeMatrixToFile(cpuTimes,
                               prefix + string("cpu") + suffix);
  Utilities::writeMatrixToFile(gpu_SoA_times,
                               prefix + string("gpu_SoA") + suffix);



  return 0;
}
