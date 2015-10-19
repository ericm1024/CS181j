// -*- C++ -*-
// Main0.cc
// cs181j hw6
// A simple example to show syntax of starting threads with std::thread,
//  and openmp

// c junk
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>

// c++ junk
#include <array>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <string>
#include <thread>

using std::string;
using std::vector;
using std::array;
using std::size_t;
using std::thread;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

// header file for openmp
#include <omp.h>

// structure to give to the thread worker function
struct WorkerData {
  unsigned int _threadIndex;
  unsigned int _numberOfThreads;
  std::string _string;
};

struct WorkerFunctor {
  unsigned int _threadIndex;
  unsigned int _numberOfThreads;
  std::string _string;

  void
  operator()() const {
    const unsigned int threadIndex = _threadIndex;
    const unsigned int numberOfThreads = _numberOfThreads;
    printf("Hello from std::thread %2u/%2u : %s\n",
           threadIndex, numberOfThreads, _string.c_str());
  }
};

// like my parents when i was in high school, we take no arguments
//int main(int argc, char* argv[]) {
int main() {

  const unsigned int numberOfThreads = 4;

  const string helloString =
    "We must carry the weight of our decisions, shepard";

  // ===============================================================
  // ********************** < do std::thread > *********************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  {

    // prepare worker assignments
    vector<WorkerFunctor> workerFunctors(numberOfThreads);
    vector<std::thread> stdThreads(numberOfThreads);
    for (unsigned int threadIndex = 0; threadIndex < numberOfThreads;
         ++threadIndex) {
      workerFunctors[threadIndex]._threadIndex = threadIndex;
      workerFunctors[threadIndex]._numberOfThreads = numberOfThreads;
      workerFunctors[threadIndex]._string = helloString;
      stdThreads[threadIndex] = std::thread(workerFunctors[threadIndex]);
    }
    // wait for the threads to finish using the receipts
    for (unsigned int threadIndex = 0; threadIndex < numberOfThreads;
         ++threadIndex) {
      stdThreads[threadIndex].join();
    }

  }
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </do std::thread > *********************
  // ===============================================================

  // ===============================================================
  // ********************** < do openmp > **************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  {

    // openmp part
    omp_set_num_threads(numberOfThreads);
#pragma omp parallel shared(helloString)
    {
      printf("Hello from openmp %2u/%2u : %s\n", omp_get_thread_num(),
             omp_get_num_threads(), helloString.c_str());
#pragma omp for
      for (int i = 0; i < 8; ++i) {
        printf("openmp thread %2u handling i = %3d\n", omp_get_thread_num(), i);
      }

    }

  }
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </do openmp > **************************
  // ===============================================================

  return 0;
}
