// -*- C++ -*-
// Main3.cc
// cs101j 2015 hw4 Problem 3
// A benchmark usage of statically-sized points for comparison to other
//  languages.

// These utilities are used on many assignments
#include "../Utilities.h"

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
// For this particular assignment, this file has the boring
//  definitions of the Point types and all the other operators that I
//  think you'll need.
#include "CommonDefinitions.h"

// Only bring in from the standard namespace things that we care about.
// Remember, it's a naughty thing to just use the whole namespace.
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

int main() {

  const std::vector<unsigned int> arrayOfNumberOfPoints =
    {100, 1000, 10000, 100000, 1000000, 10000000};
  const unsigned int numberOfTrials = 50;

  printf("number of points : elapsed time\n");

  // For each number of points
  for (const unsigned int numberOfPoints : arrayOfNumberOfPoints) {

    // Initialize the points
    std::vector<StaticPoint> points(numberOfPoints);
    for (StaticPoint & p : points) {
      p[0] = 1;
      p[1] = 1;
      p[2] = 1;
    }

    double elapsedTime = std::numeric_limits<double>::max();

    // For each trial
    for (unsigned int trialNumber = 0;
         trialNumber < numberOfTrials; ++trialNumber) {

      // Start timing
      const high_resolution_clock::time_point tic =
        high_resolution_clock::now();

      // Do some weird calculation
      double sum = 0;
      for (unsigned int i = 0; i < points.size(); ++i) {
        const StaticPoint & p = points[i];
        const StaticPoint calculated =
          p + 2 * p + 3 * (p - p) + 4 * (p + p) + 5 * (2 * p - 3 * p);
        sum += calculated[0] + calculated[1] + calculated[2];
      }

      // Stop timing
      const high_resolution_clock::time_point toc =
        high_resolution_clock::now();
      const auto dur = std::chrono::operator-(toc, tic);
      elapsedTime =
        std::min(elapsedTime,
                 duration_cast<duration<double> >(dur).count());

      // Perform a sanity check
      const double ratio = sum / points.size();
      if (std::abs(ratio - 18) > 1e-3) {
        fprintf(stderr, "incorrect answer: ratio is %lf but should be 18\n",
                ratio);
        exit(1);
      }
    }

    printf("%8.2e : %8.2e\n", float(numberOfPoints), elapsedTime);

  }

  return 0;
}
