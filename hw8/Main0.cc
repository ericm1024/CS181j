// -*- C++ -*-
// Main0.cc
// some simple tbb syntax

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

// header files for tbb
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/compat/thread>

#include <unistd.h>

using std::size_t;
using std::vector;

enum VerbosityLevel {Verbose, Quiet};

class TbbReduce {
public:

  const vector<double> * _blargs;
  const VerbosityLevel _verbosityLevel;
  unsigned int _numberOfTimesJoinWasCalled;
  unsigned int _numberOfSplitConstructions;
  unsigned int _numberOfIterations;

  TbbReduce(const vector<double> * blargs,
            const VerbosityLevel verbosityLevel) :
    _blargs(blargs),
    _verbosityLevel(verbosityLevel),
    _numberOfTimesJoinWasCalled(0),
    _numberOfSplitConstructions(0),
    _numberOfIterations(0) {
  }

  TbbReduce(const TbbReduce & other,
            tbb::split) :
    _blargs(other._blargs),
    _verbosityLevel(other._verbosityLevel),
    _numberOfTimesJoinWasCalled(0),
    _numberOfSplitConstructions(1),
    _numberOfIterations(0) {
    if (_verbosityLevel == Verbose) {
      printf("split called\n");
    }
  }

  void operator()(const tbb::blocked_range<size_t> & range) {
    if (_verbosityLevel == Verbose) {
      printf("TbbReduce asked to process range from %4zu to %4zu\n",
             range.begin(), range.end());
    }
    unsigned int temp = 0;
    const unsigned int repeats = 5000000;
    for (unsigned int i = 0; i < repeats; ++i) {
      temp = 0;
      for (unsigned int j = range.begin(); j < range.end(); ++j) {
        ++temp;
      }
    }
    _numberOfIterations += temp;
    ++_numberOfTimesJoinWasCalled;
  }

  void join(const TbbReduce & other) {
    if (_verbosityLevel == Verbose) {
      printf("join called\n");
    }
    _numberOfTimesJoinWasCalled += other._numberOfTimesJoinWasCalled;
    _numberOfSplitConstructions += other._numberOfSplitConstructions;
    _numberOfIterations += other._numberOfIterations;
  }

private:
  TbbReduce();

};

class TbbFor {
public:

  const vector<double> * _blargs;
  vector<double> * _glarbs;


  TbbFor(const vector<double> * blargs,
         vector<double> * glarbs) :
    _blargs(blargs),
    _glarbs(glarbs) {
  }

  TbbFor(const TbbFor & other) :
    _blargs(other._blargs),
    _glarbs(other._glarbs) {
    printf("copy constructor called\n");
  }

  void operator()(const tbb::blocked_range<size_t>& range) const {
    _glarbs->at(range.begin()) = 0;
    printf("TbbFor asked to process range from %4zu to %4zu\n",
           range.begin(), range.end());
  }

private:
  TbbFor();

};

//int main(int argc, char* argv[]) {
int main() {

  // we will repeat the computation for each of the numbers of threads
  vector<unsigned int> numberOfThreadsArray = {{1, 2, 4, 8}};

  for (unsigned int numberOfThreadsIndex = 0;
       numberOfThreadsIndex < numberOfThreadsArray.size();
       ++numberOfThreadsIndex) {

    const unsigned int numberOfThreads =
      numberOfThreadsArray[numberOfThreadsIndex];

    vector<double> input(100, 0);
    vector<double> output(100, 1);

    tbb::task_scheduler_init init(numberOfThreads);

    TbbFor tbbFor(&input, &output);

    parallel_for(tbb::blocked_range<size_t>(0, 100),
                 tbbFor);

  }

  printf("\n\n\n");

  for (unsigned int numberOfThreadsIndex = 0;
       numberOfThreadsIndex < numberOfThreadsArray.size();
       ++numberOfThreadsIndex) {

    const unsigned int numberOfThreads =
      numberOfThreadsArray[numberOfThreadsIndex];

    vector<double> input(100, 0);

    tbb::task_scheduler_init init(numberOfThreads);

    TbbReduce tbbReduce(&input, Verbose);

    parallel_reduce(tbb::blocked_range<size_t>(0, 100),
                    tbbReduce);
    printf("join was called %u times\n", tbbReduce._numberOfTimesJoinWasCalled);

  }

  printf("\n\n\n");

  for (unsigned int numberOfThreadsIndex = 0;
       numberOfThreadsIndex < numberOfThreadsArray.size();
       ++numberOfThreadsIndex) {

    const vector<unsigned int> grainSizes = {{1, 4, 16, 64, 256}};

    const unsigned int numberOfThreads =
      numberOfThreadsArray[numberOfThreadsIndex];

    vector<double> input(1024, 0);

    tbb::task_scheduler_init init(numberOfThreads);

    for (const unsigned int grainSize : grainSizes) {

      TbbReduce tbbReduce(&input, Quiet);

      parallel_reduce(tbb::blocked_range<size_t>(0, 1000, grainSize),
                      tbbReduce);

      printf("%u thrds grainSize %3u (%4u pieces), %3u splits, %3u joins\n",
             numberOfThreads, grainSize, 1024 / grainSize,
             tbbReduce._numberOfSplitConstructions,
             tbbReduce._numberOfTimesJoinWasCalled);
    }

  }

  return 0;
}
