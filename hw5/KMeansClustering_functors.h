// -*- C++ -*-
#ifndef KMEANS_FUNCTORS_H
#define KMEANS_FUNCTORS_H

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

#include <vector>

struct
ScalarClusterer {

  std::array<std::vector<float>, 3> _centroids;
  std::array<std::vector<float>, 3> _nextCentroids;
  std::vector<unsigned int> _nextCentroidCounts;

  ScalarClusterer(const unsigned int numberOfCentroids) :
    _nextCentroidCounts(numberOfCentroids) {
    for (unsigned int coordinate = 0; coordinate < 3; ++coordinate) {
      _centroids[coordinate].resize(numberOfCentroids);
      _nextCentroids[coordinate].resize(numberOfCentroids);
    }
  }

  void
  calculateClusterCentroids(const unsigned int numberOfIterations,
                            const std::vector<Point> & points,
                            const std::array<std::vector<float>, 3> & startingCentroids,
                            std::array<std::vector<float>, 3> * finalCentroids) {

    const unsigned int numberOfPoints = points.size();
    const unsigned int numberOfCentroids = startingCentroids[0].size();

    _centroids = startingCentroids;

    // Start with values of 0 for the next centroids
    for (unsigned int coordinate = 0; coordinate < 3; ++coordinate) {
      std::fill(_nextCentroids[coordinate].begin(),
                _nextCentroids[coordinate].end(), 0);
    }
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
        const Point firstPointsDiff =
          {{point[0] - _centroids[0][0],
            point[1] - _centroids[1][0],
            point[2] - _centroids[2][0]}};
        // First centroid is considered the closest to start
        float squaredDistanceToClosestCentroid =
          firstPointsDiff[0] * firstPointsDiff[0] +
          firstPointsDiff[1] * firstPointsDiff[1] +
          firstPointsDiff[2] * firstPointsDiff[2];
        // For each centroid after the first
        for (unsigned int centroidIndex = 1;
             centroidIndex < numberOfCentroids; ++centroidIndex) {
          const Point diff =
            {{point[0] - _centroids[0][centroidIndex],
              point[1] - _centroids[1][centroidIndex],
              point[2] - _centroids[2][centroidIndex]}};
          const float squaredDistanceToThisCentroid =
            diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
          // If we're closer, change the closest one
          if (squaredDistanceToThisCentroid < squaredDistanceToClosestCentroid) {
            indexOfClosestCentroid = centroidIndex;
            squaredDistanceToClosestCentroid = squaredDistanceToThisCentroid;
          }
        }

        // Add our point to the next centroid value
        _nextCentroids[0][indexOfClosestCentroid] += point[0];
        _nextCentroids[1][indexOfClosestCentroid] += point[1];
        _nextCentroids[2][indexOfClosestCentroid] += point[2];
        ++_nextCentroidCounts[indexOfClosestCentroid];
      }

      // Move centroids
      for (unsigned int centroidIndex = 0;
           centroidIndex < numberOfCentroids; ++centroidIndex) {
        // The next centroid value is the average of the points that were
        //  closest to it.
        _centroids[0][centroidIndex] =
          _nextCentroids[0][centroidIndex] / _nextCentroidCounts[centroidIndex];
        _centroids[1][centroidIndex] =
          _nextCentroids[1][centroidIndex] / _nextCentroidCounts[centroidIndex];
        _centroids[2][centroidIndex] =
          _nextCentroids[2][centroidIndex] / _nextCentroidCounts[centroidIndex];
        // Reset the intermediate values
        _nextCentroidCounts[centroidIndex] = 0;
        _nextCentroids[0][centroidIndex] = 0;
        _nextCentroids[1][centroidIndex] = 0;
        _nextCentroids[2][centroidIndex] = 0;
      }
    }
    *finalCentroids = _centroids;
  }

};

struct
VectorizedClusterer {

  std::array<std::vector<float>, 3> _centroids;
  std::array<std::vector<float>, 3> _nextCentroids;
  std::vector<unsigned int> _nextCentroidCounts;

  VectorizedClusterer(const unsigned int numberOfCentroids) :
    _nextCentroidCounts(numberOfCentroids) {
    for (unsigned int coordinate = 0; coordinate < 3; ++coordinate) {
      _centroids[coordinate].resize(numberOfCentroids);
      _nextCentroids[coordinate].resize(numberOfCentroids);
    }
  }

  void
  calculateClusterCentroids(const unsigned int numberOfIterations,
                            const std::vector<Point> & points,
                            const std::array<std::vector<float>, 3> & startingCentroids,
                            std::array<std::vector<float>, 3> * finalCentroids) {

    // TODO: change me

    const unsigned int numberOfPoints = points.size();
    const unsigned int numberOfCentroids = startingCentroids[0].size();

    _centroids = startingCentroids;

    // Start with values of 0 for the next centroids
    for (unsigned int coordinate = 0; coordinate < 3; ++coordinate) {
      std::fill(_nextCentroids[coordinate].begin(),
                _nextCentroids[coordinate].end(), 0);
    }
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
        const Point firstPointsDiff =
          {{point[0] - _centroids[0][0],
            point[1] - _centroids[1][0],
            point[2] - _centroids[2][0]}};
        // First centroid is considered the closest to start
        float squaredDistanceToClosestCentroid =
          firstPointsDiff[0] * firstPointsDiff[0] +
          firstPointsDiff[1] * firstPointsDiff[1] +
          firstPointsDiff[2] * firstPointsDiff[2];
        // For each centroid after the first
        for (unsigned int centroidIndex = 1;
             centroidIndex < numberOfCentroids; ++centroidIndex) {
          const Point diff =
            {{point[0] - _centroids[0][centroidIndex],
              point[1] - _centroids[1][centroidIndex],
              point[2] - _centroids[2][centroidIndex]}};
          const float squaredDistanceToThisCentroid =
            diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
          // If we're closer, change the closest one
          if (squaredDistanceToThisCentroid < squaredDistanceToClosestCentroid) {
            indexOfClosestCentroid = centroidIndex;
            squaredDistanceToClosestCentroid = squaredDistanceToThisCentroid;
          }
        }

        // Add our point to the next centroid value
        _nextCentroids[0][indexOfClosestCentroid] += point[0];
        _nextCentroids[1][indexOfClosestCentroid] += point[1];
        _nextCentroids[2][indexOfClosestCentroid] += point[2];
        ++_nextCentroidCounts[indexOfClosestCentroid];
      }

      // Move centroids
      for (unsigned int centroidIndex = 0;
           centroidIndex < numberOfCentroids; ++centroidIndex) {
        // The next centroid value is the average of the points that were
        //  closest to it.
        _centroids[0][centroidIndex] =
          _nextCentroids[0][centroidIndex] / _nextCentroidCounts[centroidIndex];
        _centroids[1][centroidIndex] =
          _nextCentroids[1][centroidIndex] / _nextCentroidCounts[centroidIndex];
        _centroids[2][centroidIndex] =
          _nextCentroids[2][centroidIndex] / _nextCentroidCounts[centroidIndex];
        // Reset the intermediate values
        _nextCentroidCounts[centroidIndex] = 0;
        _nextCentroids[0][centroidIndex] = 0;
        _nextCentroids[1][centroidIndex] = 0;
        _nextCentroids[2][centroidIndex] = 0;
      }
    }
    *finalCentroids = _centroids;
  }

};

#endif // KMEANS_FUNCTORS_H
