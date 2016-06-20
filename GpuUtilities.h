// -*- C++ -*-
#ifndef GPU_UTILITIES_H
#define GPU_UTILITIES_H

#include <string>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <time.h>

// stolen from http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define checkCudaError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline
void
gpuAssert(const cudaError_t code, const char *file, const int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPU Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort == true) {
      exit(code);
    }
  }
}

// included here because dev_mem depends on checkCudaError
#include "dev_mem.hpp"

// This is a little utility function that can be used to suppress any
//  compiler warnings about unused variables.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
template <class T>
void ignoreUnusedVariable(T & t) {
}
#pragma GCC diagnostic pop

namespace TimeUtility {

typedef timespec   PreCpp11TimePoint;

double
getElapsedTime(const PreCpp11TimePoint start,
               const PreCpp11TimePoint end) {
  PreCpp11TimePoint temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return double(temp.tv_sec) + double(temp.tv_nsec) / 1e9;
}

PreCpp11TimePoint
getCurrentTime() {
  PreCpp11TimePoint tic;
  clock_gettime(CLOCK_MONOTONIC, &tic);
  return tic;
}

}

namespace GpuUtilities {

__global__
void
countThreads_kernel(unsigned int * totalCount) {
  atomicAdd(totalCount, 1);
}

void
resetGpu() {
  // allocate somewhere for the threads to count
  unsigned int *dev_junkDataCounter;
  checkCudaError(cudaMalloc((void **) &dev_junkDataCounter,
                            1*sizeof(unsigned int)));

  const unsigned int resetGpuNumberOfThreadsPerBlock = 1024;
  const unsigned int resetGpuNumberOfBlocks = 1e8 /
    resetGpuNumberOfThreadsPerBlock;
  countThreads_kernel<<<resetGpuNumberOfBlocks,
    resetGpuNumberOfThreadsPerBlock>>>(dev_junkDataCounter);

  // pull the junk data counter back from the device, for fun.
  unsigned int junkDataCounter;
  checkCudaError(cudaMemcpy(&junkDataCounter, dev_junkDataCounter,
                            1*sizeof(unsigned int),
                            cudaMemcpyDeviceToHost));
  volatile unsigned int deOptimizer = junkDataCounter;
  ignoreUnusedVariable(deOptimizer);

  // clean up
  checkCudaError(cudaFree(dev_junkDataCounter));
}

}

// computes p/q of an integral type, rounding up
// NB: this probably only works with positive numbers when p + (q-1) doesn't
// overflow
template <typename T1, typename T2>
__host__ __device__
typename std::enable_if<std::is_integral<T1>::value
                        && std::is_integral<T2>::value
                        , typename std::common_type<T1, T2>::type
                        >::type
div_round_up(T1 p, T2 q)
{
        return (p + (q - 1))/q;
}

// ugh, because we can't call library functions in __device__code
template <typename T>
__host__ __device__
static typename std::enable_if<std::is_arithmetic<T>::value, T>::type
min(const T lhs, const T rhs)
{
        return lhs < rhs ? lhs : rhs;
}

template <typename T>
__host__ __device__
static typename std::enable_if<std::is_arithmetic<T>::value, T>::type
max(const T lhs, const T rhs)
{
        return lhs < rhs ? rhs : lhs;
}

#endif // GPU_UTILITIES_H
