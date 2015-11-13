// -*- C++ -*-
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <ctime>
#include <stdint.h>
#include <cfloat>

// All of the cuda functions are defined here.
#include <cuda_runtime.h>

// These utilities are used on the GPU assignments
#include "../GpuUtilities.h"

// I suppose we don't actually have to include this, but it's good practice.
#include "Main4_cuda.cuh"

void
runGpuTimingTest(const unsigned int numberOfTrials,
                 const unsigned int maxNumberOfBlocks,
                 const unsigned int numberOfThreadsPerBlock,
                 double * elapsedTime) {

  // TODO: allocate device-side inputs

  // TODO: copy host inputs to device

  // TODO: allocate device-side outputs

  // TODO: calculate the number of blocks

  *elapsedTime = DBL_MAX; // sigh, no numeric_limits

  // run the test repeatedly
  for (unsigned int trialNumber = 0;
       trialNumber < numberOfTrials; ++trialNumber) {

    // this forces the GPU to run another kernel, kind of like
    //  "resetting the cache" for the cpu versions.
    GpuUtilities::resetGpu();

    // start timing
    const TimeUtility::PreCpp11TimePoint tic = TimeUtility::getCurrentTime();

    // TODO: run kernel

    // see if there was an error in the kernel launch
    checkCudaError(cudaPeekAtLastError());

    // wait for the kernel to stop
    checkCudaError(cudaDeviceSynchronize());

    // stop timing
    const TimeUtility::PreCpp11TimePoint toc = TimeUtility::getCurrentTime();
    const double thisTrialsElapsedTime =
      TimeUtility::getElapsedTime(tic, toc);
    *elapsedTime = std::min(*elapsedTime, thisTrialsElapsedTime);
  }

  // TODO: copy device outputs back to host

  // TODO: free memory
}
