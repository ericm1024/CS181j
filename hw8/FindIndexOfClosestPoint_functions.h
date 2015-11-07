// -*- C++ -*-
#ifndef FIND_INDEX_OF_CLOSEST_POINT_FUNCTIONS_HW8_H
#define FIND_INDEX_OF_CLOSEST_POINT_FUNCTIONS_HW8_H

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

// These utilities are used on many assignments
#include "../Utilities.h"

// include versions from last assignment
#include "../hw7/FindIndexOfClosestPoint_functions.h"

// header files for tbb
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

unsigned int
findIndexOfClosestPoint_tbb(const unsigned int numberOfThreads,
                            const std::vector<Point> & points,
                            const Point & searchLocation) {
  // Ignore this number of threads, I set it in the main function
  ignoreUnusedVariable(numberOfThreads);

  // TODO

  return 0;
}

#endif // FIND_INDEX_OF_CLOSEST_POINT_FUNCTIONS_HW8_H
