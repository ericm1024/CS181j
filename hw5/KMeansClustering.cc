// -*- C++ -*-
// KMeansClustering.cc
// cs181j 2015 hw5 Problem 3
// An exercise in vectorizing, accelerating the computation of k-means
//  clustering.

// These utilities are used on many assignments
#include "../Utilities.h"

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

#include "KMeansClustering_functors.h"

using std::vector;
using std::array;
using std::string;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

void
checkResult(const array<vector<float>, 3> & correctResult,
            const array<vector<float>, 3> & testResult,
            const string & testName,
            const double absoluteErrorTolerance) {
  char sprintfBuffer[500];
  if (correctResult[0].size() != testResult[0].size()) {
    sprintf(sprintfBuffer, "test result has the wrong number of entries: %zu "
            "instead of %zu, test named "
            BOLD_ON FG_RED "%s" RESET "\n",
            testResult[0].size(), correctResult[0].size(),
            testName.c_str());
    throw std::runtime_error(sprintfBuffer);
  }
  for (size_t i = 0; i < correctResult[0].size(); ++i) {
    double absoluteError = 0;
    for (unsigned int coordinate = 0; coordinate < 3; ++coordinate) {
      absoluteError +=
        (correctResult[coordinate][i] - testResult[coordinate][i]) *
        (correctResult[coordinate][i] - testResult[coordinate][i]);
    }
    absoluteError = std::sqrt(absoluteError);
    if (absoluteError > absoluteErrorTolerance) {
      sprintf(sprintfBuffer, "wrong result for centroid number %zu in test result, "
              "it's (%e, %e, %e) but should be (%e, %e, %e), test named "
              BOLD_ON FG_RED "%s" RESET "\n", i,
              testResult[0][i], testResult[1][i], testResult[2][i],
              correctResult[0][i], correctResult[1][i], correctResult[2][i],
              testName.c_str());
      throw std::runtime_error(sprintfBuffer);
    }
  }
}

template <class Clusterer>
void
runTimingTest(const unsigned int numberOfTrials,
              const unsigned int numberOfIterations,
              const vector<Point> & points,
              const array<vector<float>, 3> & startingCentroids,
              Clusterer * clusterer,
              array<vector<float>, 3> * finalCentroids,
              double * elapsedTime) {

  *elapsedTime = std::numeric_limits<double>::max();

  for (unsigned int trialNumber = 0;
       trialNumber < numberOfTrials; ++trialNumber) {

    // Reset the final centroids
    (*finalCentroids) = startingCentroids;

    // Start measuring
    const high_resolution_clock::time_point tic = high_resolution_clock::now();

    // Do the clustering
    clusterer->calculateClusterCentroids(numberOfIterations,
                                         points,
                                         startingCentroids,
                                         finalCentroids);

    // Stop measuring
    const high_resolution_clock::time_point toc = high_resolution_clock::now();
    const double thisTrialsElapsedTime =
      duration_cast<duration<double> >(toc - tic).count();
    // Take the minimum values from all trials
    *elapsedTime = std::min(*elapsedTime, thisTrialsElapsedTime);
  }

}

int main() {

  // ===========================================================================
  // *************************** < Inputs> *************************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  // A lot of homeworks will run something over a range of sizes,
  //  which will then be plotted by some script.
  // This controls how many points are used.
  const array<double, 2> rangeOfNumberOfPoints = {{1e3, 1e5}};
  // This number controls how many data points are made and plotted.
  const unsigned int numberOfPointsDataPoints = 5;
  // This controls how many centroids are used.
  const array<double, 2> rangeOfNumberOfCentroids = {{5, 200}};
  // This number controls how many data points are made and plotted.
  const unsigned int numberOfCentroidsDataPoints = 10;
  // In real k-means calculations, the centroid updates would happen
  //  until some condition is satisfied.  In this, we'll just iterate
  //  a fixed number of times, so that all methods do the same amount
  //  of work.
  const unsigned int numberOfIterations = 5;
  // This is the standard number of times the calculation is repeated.
  const unsigned int numberOfTrials = 10;

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // *************************** </Inputs> *************************************
  // ===========================================================================

  // On each test, we need to make sure we get the same result.  A test will
  //  fail if the difference between any entry in our result is more than
  //  absoluteErrorTolerance different than entries we got with another method.
  const double absoluteErrorTolerance = 1e-4;

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
    scalarTimes(numberOfPointsDataPoints,
                vector<double>(numberOfCentroidsDataPoints, 0));
  vector<vector<double> >
    vectorizedTimes(numberOfPointsDataPoints,
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

    const high_resolution_clock::time_point thisSizesTic=
      high_resolution_clock::now();

    // For each number of centroids data point
    for (unsigned int numberOfCentroidsDataPointIndex = 0;
         numberOfCentroidsDataPointIndex < numberOfCentroidsDataPoints;
         ++numberOfCentroidsDataPointIndex) {
      // Calculate the number of centroids so that it's linear on a
      //  log scale.
      const size_t numberOfCentroids =
        Utilities::interpolateNumberLinearlyOnLogScale(rangeOfNumberOfCentroids[0],
                                                       rangeOfNumberOfCentroids[1],
                                                       numberOfCentroidsDataPoints,
                                                       numberOfCentroidsDataPointIndex);

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
      array<vector<float>, 3> startingCentroids;
      startingCentroids[0].resize(numberOfCentroids);
      startingCentroids[1].resize(numberOfCentroids);
      startingCentroids[2].resize(numberOfCentroids);
      for (unsigned int centroidIndex = 0;
           centroidIndex < numberOfCentroids; ++centroidIndex) {
        startingCentroids[0][centroidIndex] =
          uniformRealDistribution(randomNumberGenerator);
        startingCentroids[1][centroidIndex] =
          uniformRealDistribution(randomNumberGenerator);
        startingCentroids[2][centroidIndex] =
          uniformRealDistribution(randomNumberGenerator);
      }

      ScalarClusterer scalarClusterer(numberOfCentroids);

      array<vector<float>, 3> scalarFinalCentroids;
      scalarFinalCentroids[0].resize(numberOfCentroids);
      scalarFinalCentroids[1].resize(numberOfCentroids);
      scalarFinalCentroids[2].resize(numberOfCentroids);
      double scalarElapsedTime;
      runTimingTest(numberOfTrials,
                    numberOfIterations,
                    points,
                    startingCentroids,
                    &scalarClusterer,
                    &scalarFinalCentroids,
                    &scalarElapsedTime);

      VectorizedClusterer vectorizedClusterer(numberOfCentroids);

      array<vector<float>, 3> vectorizedFinalCentroids;
      vectorizedFinalCentroids[0].resize(numberOfCentroids);
      vectorizedFinalCentroids[1].resize(numberOfCentroids);
      vectorizedFinalCentroids[2].resize(numberOfCentroids);
      double vectorizedElapsedTime;
      runTimingTest(numberOfTrials,
                    numberOfIterations,
                    points,
                    startingCentroids,
                    &vectorizedClusterer,
                    &vectorizedFinalCentroids,
                    &vectorizedElapsedTime);
      checkResult(scalarFinalCentroids,
                  vectorizedFinalCentroids,
                  string("vectorized"),
                  absoluteErrorTolerance);

      numberOfPointsMatrixForPlotting[numberOfPointsDataPointIndex][numberOfCentroidsDataPointIndex] =
        numberOfPoints;
      numberOfCentroidsMatrixForPlotting[numberOfPointsDataPointIndex][numberOfCentroidsDataPointIndex] =
        numberOfCentroids;
      scalarTimes[numberOfPointsDataPointIndex][numberOfCentroidsDataPointIndex] =
        scalarElapsedTime;
      vectorizedTimes[numberOfPointsDataPointIndex][numberOfCentroidsDataPointIndex] =
        vectorizedElapsedTime;

    }
    const high_resolution_clock::time_point thisSizesToc =
      high_resolution_clock::now();
    const double thisSizesElapsedTime =
      duration_cast<duration<double> >(thisSizesToc -
                                       thisSizesTic).count();
    printf("processing %8.2e points took %7.2f seconds\n",
           double(numberOfPoints),
           thisSizesElapsedTime);
  }

  const string prefix = "data/KMeansClustering_";
  const string suffix = "_shuffler";

  Utilities::writeMatrixToFile(numberOfPointsMatrixForPlotting,
                               prefix + string("numberOfPoints") + suffix);
  Utilities::writeMatrixToFile(numberOfCentroidsMatrixForPlotting,
                               prefix + string("numberOfCentroids") + suffix);
  Utilities::writeMatrixToFile(scalarTimes,
                               prefix + string("scalar") + suffix);
  Utilities::writeMatrixToFile(vectorizedTimes,
                               prefix + string("vectorized") + suffix);

  return 0;
}
