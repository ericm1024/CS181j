// -*- C++ -*-
// Main0.cc
// cs181j hw1 example
// A simple file to illustrate how to use PAPI to measure things
// Jeff Amelang, 2015

// c junk
#include <cstdio>
#include <cstdlib>
#include <cmath>

// c++ junk
#include <array>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>

// All the magic for measuring cache misses is in papi
#include <papi.h>

// A very thin wrapper around PAPI_strerror.
void
handlePapiError(const int papiReturnVal) {
  if (papiReturnVal != PAPI_OK) {
    fprintf(stderr, "PAPI error: %s\n",
           PAPI_strerror(papiReturnVal));
    exit(1);
  }
}

// Reduce the amount of typing we have to do for timing things.
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

int main() {

  // ===========================================================================
  // *************************** < Papi initialization> ************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  // Initialize the papi library
  PAPI_library_init(PAPI_VER_CURRENT);
  // Initialize an event set
  int papiEventSet = PAPI_NULL;
  handlePapiError(PAPI_create_eventset(&papiEventSet));

  // Add number of floating point divides
  handlePapiError(PAPI_add_event(papiEventSet, PAPI_FDV_INS));

  // Tell papi to start, after which we can do timing
  handlePapiError(PAPI_start(papiEventSet));

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // *************************** </Papi initialization> ************************
  // ===========================================================================

  // When we tic and toc with papi, we have to have somewhere for papi
  //  to put its result.
  // This variable is where it'll store the result.
  // It's an array because we could record more than one event.
  long long papiCounters[1];

  // Create a c++11 random number generator
  std::mt19937 randomNumberEngine;
  std::uniform_real_distribution<double> randomNumberGenerator(0, 1);

  // Start timing
  const high_resolution_clock::time_point tic =
    high_resolution_clock::now();
  handlePapiError(PAPI_accum(papiEventSet, papiCounters));

  // We'll record how many floating point divides are used to generate a
  //  bunch of random numbers, because why not?
  const size_t numberOfRandomNumbers = 1e8;
  double randomNumberSum = 0;
  for (size_t i = 0; i < numberOfRandomNumbers; ++i) {
    const double randomNumberBetween0And1 =
      randomNumberGenerator(randomNumberEngine);
    randomNumberSum += randomNumberBetween0And1;
  }

  // Stop timing
  const high_resolution_clock::time_point toc =
    high_resolution_clock::now();
  papiCounters[0] = 0;
  handlePapiError(PAPI_accum(papiEventSet, papiCounters));
  const double elapsedTime =
    duration_cast<duration<double> >(toc - tic).count();
  const size_t numberOfDivisions = papiCounters[0];

  printf("randomNumberSum was %11.4e, which is %%%5.1f of the number of random "
         "numbers (%8.2e)\n", randomNumberSum,
         100. * randomNumberSum / float(numberOfRandomNumbers),
         float(numberOfRandomNumbers));
  printf("in %5.2lf seconds, measured %zu (%8.2e) divisions used in the "
         "generation of the %zu (%8.2e) random numbers\n", elapsedTime,
         numberOfDivisions, float(numberOfDivisions),
         numberOfRandomNumbers, float(numberOfRandomNumbers));

  // ===========================================================================
  // *************************** < Papi cleanup> *******************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  handlePapiError(PAPI_stop(papiEventSet, 0));
  handlePapiError(PAPI_cleanup_eventset(papiEventSet));
  handlePapiError(PAPI_destroy_eventset(&papiEventSet));

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // *************************** </Papi cleanup> *******************************
  // ===========================================================================

  return 0;
}
