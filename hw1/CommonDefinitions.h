// -*- C++ -*-
#ifndef COMMON_DEFINITIONS_H
#define COMMON_DEFINITIONS_H

// c junk
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdio>

// c++ junk
#include <array>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <random>
#include <fstream>

// This is a little utility function that can be used to suppress any
//  compiler warnings about unused variables.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
template <class T>
void ignoreUnusedVariable(T & t) {
}
#pragma GCC diagnostic pop

// This is a little utility macro that can be useful for debugging.
#define debug(s, ...)                                           \
  do {                                                          \
    fprintf (stderr, "(%-20s:%40s:%4d) -- " s "\n",             \
             __FILE__, __func__, __LINE__, ##__VA_ARGS__);      \
    fflush (stderr);                                            \
  } while (0)

namespace Utilities {

void
verifyThatDirectoryExists(const std::string & path) {
  std::ifstream test(path);
  if ((bool)test == false) {
    fprintf(stderr, "Error, cannot find directory at %s.  "
            "Please make it yourself (\"mkdir %s\")\n",
            path.c_str(), path.c_str());
    exit(1);
  }
}

size_t
interpolateNumberLinearlyOnLogScale(const size_t lower,
                                    const size_t upper,
                                    const unsigned int numberOfPoints,
                                    const unsigned int pointIndex) {
  const double percent =
    pointIndex / double(numberOfPoints - 1);
  const double power = std::log10(lower) +
    percent * (std::log10(upper) - std::log10(lower));
  return std::pow(10., power);
}

void
clearCpuCache() {

  volatile double uselessJunkSum = 0;
  const size_t sizeOfUselessJunk = 1e7;
  std::vector<double> uselessJunk(sizeOfUselessJunk, 1);
  std::accumulate(uselessJunk.begin(), uselessJunk.end(), uselessJunkSum);

}

}

#endif // COMMON_DEFINITIONS_H
