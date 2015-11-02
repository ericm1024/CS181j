// -*- C++ -*-
#ifndef KMEANS_FUNCTORS_H
#define KMEANS_FUNCTORS_H

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

#include <vector>

#include <omp.h>

struct
SerialClusterer {

  std::vector<Point> _centroids;
  std::vector<Point> _nextCentroids;
  std::vector<unsigned int> _nextCentroidCounts;

  SerialClusterer(const unsigned int numberOfCentroids) :
    _centroids(numberOfCentroids),
    _nextCentroids(numberOfCentroids),
    _nextCentroidCounts(numberOfCentroids) {
  }

  void
  calculateClusterCentroids(const unsigned int ignoredNumberOfThreads,
                            const unsigned int numberOfIterations,
                            const std::vector<Point> & points,
                            const std::vector<Point> & startingCentroids,
                            std::vector<Point> * finalCentroids) {
    ignoreUnusedVariable(ignoredNumberOfThreads);

    const unsigned int numberOfPoints = points.size();
    const unsigned int numberOfCentroids = startingCentroids.size();

    _centroids = startingCentroids;

    // Start with values of 0 for the next centroids
    std::fill(_nextCentroids.begin(), _nextCentroids.end(),
              (Point) {{0., 0., 0.}});
    std::fill(_nextCentroidCounts.begin(), _nextCentroidCounts.end(), 0);

    // For each of a fixed number of iterations
    for (unsigned int iterationNumber = 0;
         iterationNumber < numberOfIterations; ++iterationNumber) {
      // Calculate next centroids
      for (unsigned int pointIndex = 0;
           pointIndex < numberOfPoints; ++pointIndex) {
        const Point & point = points[pointIndex];
        // Find which centroid this point is closest to
        unsigned int indexOfClosestCentroid = 0;
        // First centroid is considered the closest to start
        double squaredDistanceToClosestCentroid =
          squaredMagnitude(point - _centroids[0]);
        // For each centroid after the first
        for (unsigned int centroidIndex = 1;
             centroidIndex < numberOfCentroids; ++centroidIndex) {
          const double squaredDistanceToThisCentroid =
            squaredMagnitude(point - _centroids[centroidIndex]);
          // If we're closer, change the closest one
          if (squaredDistanceToThisCentroid < squaredDistanceToClosestCentroid) {
            indexOfClosestCentroid = centroidIndex;
            squaredDistanceToClosestCentroid = squaredDistanceToThisCentroid;
          }
        }

        // Add our point to the next centroid value
        _nextCentroids[indexOfClosestCentroid] += point;
        ++_nextCentroidCounts[indexOfClosestCentroid];
      }

      // Move centroids
      for (unsigned int centroidIndex = 0;
           centroidIndex < numberOfCentroids; ++centroidIndex) {
        // The next centroid value is the average of the points that were
        //  closest to it.
        _centroids[centroidIndex] =
          _nextCentroids[centroidIndex] / _nextCentroidCounts[centroidIndex];
        // Reset the intermediate values
        _nextCentroidCounts[centroidIndex] = 0;
        _nextCentroids[centroidIndex] = (Point) {{0., 0., 0.}};
      }
    }
    *finalCentroids = _centroids;
  }

};

struct
ThreadedClusterer {

  std::vector<Point> _centroids;
  std::vector<Point> _nextCentroids;
  std::vector<unsigned int> _nextCentroidCounts;

  ThreadedClusterer(const unsigned int numberOfCentroids) :
    _centroids(numberOfCentroids),
    _nextCentroids(numberOfCentroids),
    _nextCentroidCounts(numberOfCentroids) {
  }

  void
  calculateClusterCentroids(const unsigned int numberOfThreads,
                            const unsigned int numberOfIterations,
                            const std::vector<Point> & points,
                            const std::vector<Point> & startingCentroids,
                            std::vector<Point> * finalCentroids) {

    omp_set_num_threads(numberOfThreads);

    // TODO: make a threaded version
    throw std::runtime_error("threaded version not yet implemented");

  }

};

#endif // KMEANS_FUNCTORS_H
