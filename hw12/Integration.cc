// -*- C++ -*-
// Integration.cc
// cs181j hw12
// An example to illustrate how to use mpi to parallelize a scalar
// function integrator.

// magic header for all mpi stuff
#include <mpi.h>

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

// These utilities are used on many assignments
#include "../Utilities.h"

using std::vector;
using std::array;

// magic header for all mpi stuff
#include <mpi.h>

template <class ScalarFunction>
double
calculateIntegral(ScalarFunction scalarFunction,
                  const array<double, 2> integrationBounds,
                  const size_t numberOfIntervals) {
  double sum = 0;
  const double dx =
    (integrationBounds[1] - integrationBounds[0]) / numberOfIntervals;
  for (size_t intervalIndex = 0;
       intervalIndex < numberOfIntervals; ++intervalIndex) {
    const double evaluationPoint =
      integrationBounds[0] + (intervalIndex + 0.5) * dx;
    sum += scalarFunction(evaluationPoint);
  }
  sum *= dx;
  return sum;
}

int main(int argc, char* argv[]) {

  // ===============================================================
  // ********************** < Input> *******************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  // the program will calculate the integral using various number of intervals.
  // you should see the answer get more accurate with more intervals.
  // instead of specify the number of intervals, specify the order of magnitude.
  // as in, do 10^1, 10^3, etc intervals
  const vector<double> intervalOrdersOfMagnitude = {{1, 3, 4, 5, 6, 7, 8, 9}};
  // this is the range over which to integrate the function
  const array<double, 2> integrationBounds = {{.61, 1.314}};

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </Input> *******************************
  // ===============================================================

  // TODO: Initialize MPI

  // TODO: Figure out what rank I am
  const unsigned int numberOfProcesses = 1; // TODO: change me please

  const double integrationRangeWidth =
    integrationBounds[1] - integrationBounds[0];

  // for each number of intervals
  for (unsigned int sizeIndex = 0;
       sizeIndex < intervalOrdersOfMagnitude.size();
       sizeIndex++) {
    // compute the number of intervals for this order of magnitude
    const size_t numberOfIntervals =
      size_t(pow(10.0, intervalOrdersOfMagnitude[sizeIndex]));

    const double tic = MPI_Wtime();
    // do the calculation in the helper function.
    // note that we're passing the function to integrate as a parameter.
    const double thisRanksIntegral =
      calculateIntegral(sin, integrationBounds,
                        numberOfIntervals);

    const double elapsedTime = MPI_Wtime() - tic;

    const double correctIntegral =
      cos(integrationBounds[0]) - cos(integrationBounds[1]);
    const double error = std::abs(thisRanksIntegral - correctIntegral);
    const double relativeError = error / std::abs(correctIntegral);
    if (relativeError > 1e-3) {
      fprintf(stderr, "error for %zu intervals, relative error is %e\n",
              numberOfIntervals, relativeError);
      exit(1);
    }

    printf("%3u processes, 10^%4.2f intervals, relative error %8.2e in %12.8lf "
           "seconds\n",
           numberOfProcesses,
           intervalOrdersOfMagnitude[sizeIndex],
           relativeError,
           elapsedTime);

  }

  return 0;
}
