// -*- C++ -*-
#ifndef FIND_INDEX_OF_CLOSEST_POINT_FUNCTIONS_H
#define FIND_INDEX_OF_CLOSEST_POINT_FUNCTIONS_H

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

// These utilities are used on many assignments
#include "../Utilities.h"

// header file for openmp
#include <omp.h>

unsigned int
findIndexOfClosestPoint_serial(const unsigned int ignoredNumberOfThreads,
                               const std::vector<Point> & points,
                               const Point & searchLocation) {
  ignoreUnusedVariable(ignoredNumberOfThreads);

  const unsigned int numberOfPoints = points.size();

  unsigned int indexOfClosestPoint = 0;
  double squaredMagnitudeOfClosestPoint =
    squaredMagnitude(points[0] - searchLocation);
  for (unsigned int pointIndex = 1;
       pointIndex < numberOfPoints; ++pointIndex) {
    const double squaredMagnitudeToThisPoint =
      squaredMagnitude(points[pointIndex] - searchLocation);
    if (squaredMagnitudeToThisPoint < squaredMagnitudeOfClosestPoint) {
      indexOfClosestPoint = pointIndex;
      squaredMagnitudeOfClosestPoint = squaredMagnitudeToThisPoint;
    }
  }

  return indexOfClosestPoint;
}

unsigned int
findIndexOfClosestPoint_threaded(const unsigned int numberOfThreads,
                                 const std::vector<Point> & points,
                                 const Point & searchLocation) {
        omp_set_num_threads(numberOfThreads);

        const unsigned int numberOfPoints = points.size();

        unsigned int midx = 0;
        double min_dist = squaredMagnitude(points[midx] - searchLocation);

#pragma omp parallel shared(midx, min_dist)
        {
                auto this_thread_midx = midx;
                auto this_thread_min_dist = min_dist;

#pragma omp for
                for (auto i = 0u; i < numberOfPoints; ++i) {
                        const double squaredMagnitudeToThisPoint =
                                squaredMagnitude(points[i] - searchLocation);
                        if (squaredMagnitudeToThisPoint < this_thread_min_dist) {
                                this_thread_midx = i;
                                this_thread_min_dist = squaredMagnitudeToThisPoint;
                        }
                }

#pragma omp critical
                if (this_thread_min_dist < min_dist) {
                        min_dist = this_thread_min_dist;
                        midx = this_thread_midx;
                }
        }

        return midx;
}

#endif // FIND_INDEX_OF_CLOSEST_POINT_FUNCTIONS_H
