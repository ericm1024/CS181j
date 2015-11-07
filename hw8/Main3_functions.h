// -*- C++ -*-
// Main3_functions.cc
// cs181j hw7 Problem 3
// These are the functors for the Histogram executable.

// These utilities are used on many assignments
#include "../Utilities.h"

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

// header file for openmp
#include <omp.h>

// header files for tbb
#include <tbb/task_scheduler_init.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>
#include <tbb/mutex.h>
#include <tbb/spin_mutex.h>
#include <tbb/queuing_mutex.h>

double
functionToHistogram(const double x) {
  return 0.51 * (0.500 * std::sin(1e0 * x) + 0.325 * std::sin(1e1 * x) +
                 0.100 * std::sin(1e2 * x) + 0.050 * std::sin(1e3 * x) + 0.961);
}

void
calculateHistogram_serial(const std::vector<double> & input,
                          const unsigned int lockBucketSize,
                          std::vector<unsigned int> * histogram) {
  // The lockBucketSize is here so that I can use the same testing logic
  //  for all methods.  Do NOT use it here.  There shouldn't be locks in this
  //  version.
  ignoreUnusedVariable(lockBucketSize);

  const unsigned int numberOfBuckets = histogram->size();
  const unsigned int inputSize = input.size();
  const double bucketSize = 1./double(numberOfBuckets);
  std::vector<unsigned int> & histogramReference = *histogram;
  for (unsigned int index = 0; index < inputSize; ++index) {
    const unsigned int bucketIndex = functionToHistogram(input[index]) / bucketSize;
    ++histogramReference[bucketIndex];
  }
}

void
calculateHistogram_reduction(const std::vector<double> & input,
                             const unsigned int lockBucketSize,
                             std::vector<unsigned int> * histogram) {
  // The lockBucketSize is here so that I can use the same testing logic
  //  for all methods.  Do NOT use it here.  There shouldn't be locks in this
  //  version.
  ignoreUnusedVariable(lockBucketSize);

  // TODO: replace this with an openmp reduction version
  calculateHistogram_serial(input,
                            lockBucketSize,
                            histogram);

}

void
calculateHistogram_atomics(const std::vector<double> & input,
                           const unsigned int lockBucketSize,
                           std::vector<unsigned int> * histogram) {
  // The lockBucketSize is here so that I can use the same testing logic
  //  for all methods.  Do NOT use it here.  There shouldn't be locks in this
  //  version.
  ignoreUnusedVariable(lockBucketSize);

  // TODO: replace this with an openmp atomics version
  calculateHistogram_serial(input,
                            lockBucketSize,
                            histogram);

}

void
calculateHistogram_atomicFlagLocks(const std::vector<double> & input,
                                   const unsigned int lockBucketSize,
                                   std::vector<unsigned int> * histogram) {

  // TODO: replace this with an openmp locks version using atomic_flag locks
  calculateHistogram_serial(input,
                            lockBucketSize,
                            histogram);

#if 0
  // locking example
  std::atomic_flag lock;
  while (locks[lockIndex].test_and_set(std::memory_order_acquire));
  // do neat stuff
  locks[lockIndex].clear(std::memory_order_release);
#endif
}

void
calculateHistogram_tbbReduction(const std::vector<double> & input,
                                const unsigned int lockBucketSize,
                                std::vector<unsigned int> * histogram) {
  // The lockBucketSize is here so that I can use the same testing logic
  //  for all methods.  Do NOT use it here.  There shouldn't be locks in this
  //  version.
  ignoreUnusedVariable(lockBucketSize);

  // TODO: replace this with a tbb reduction version
  calculateHistogram_serial(input,
                            lockBucketSize,
                            histogram);

}

void
calculateHistogram_tbbAtomics(const std::vector<double> & input,
                              const unsigned int lockBucketSize,
                              std::vector<unsigned int> * histogram) {
  // The lockBucketSize is here so that I can use the same testing logic
  //  for all methods.  Do NOT use it here.  There shouldn't be locks in this
  //  version.
  ignoreUnusedVariable(lockBucketSize);

  // TODO: replace this with a tbb atomics version
  calculateHistogram_serial(input,
                            lockBucketSize,
                            histogram);

}

void
calculateHistogram_tbbLocks(const std::vector<double> & input,
                            const unsigned int lockBucketSize,
                            std::vector<unsigned int> * histogram) {

  // TODO: replace this with a tbb locks version using speculative_spin_mutex
  //  locks
  calculateHistogram_serial(input,
                            lockBucketSize,
                            histogram);


  // locking example
#if 0
  tbb::speculative_spin_mutex lock;
  {
    // stuff to not protect
    typename tbb::speculative_spin_mutex::scoped_lock scopedLock(lock);
    // stuff to protect
  }
#endif
}
