// -*- C++ -*-
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <ctime>
#include <cfloat>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "ReductionRace_cuda.cuh"
#include "../GpuUtilities.h"

__device__
unsigned int
getLog2OfPowerOf2(unsigned int integerThatIsAPowerOf2) {
  unsigned int log2 = 0;
  while (integerThatIsAPowerOf2 >>= 1) {
    ++log2;
  }
  return log2;
}

__global__
void
doReduction_serialBlockReduction(const unsigned int * const input,
                                 const unsigned int numberOfInputs,
                                 unsigned int * const output) {
  // TODO
}

__global__
void
doReduction_stupendous(const unsigned int * const input,
                       const unsigned int numberOfInputs,
                       unsigned int * const output) {
  // TODO
}

void
runGpuTimingTest(const unsigned int numberOfTrials,
                 const unsigned int maxNumberOfBlocks,
                 const unsigned int numberOfThreadsPerBlock,
                 const CudaReductionStyle cudaReductionStyle,
                 const unsigned int * const input,
                 const unsigned int numberOfInputs,
                 unsigned int * const output,
                 double * const elapsedTime) {

  // allocate device-side input
  unsigned int *dev_input;
  checkCudaError(cudaMalloc((void **) &dev_input,
                            numberOfInputs*sizeof(unsigned int)));
  checkCudaError(cudaMemcpy(dev_input, input,
                            numberOfInputs*sizeof(unsigned int),
                            cudaMemcpyHostToDevice));

  // allocate device-side output
  unsigned int *dev_output;
  checkCudaError(cudaMalloc((void **) &dev_output, 1*sizeof(unsigned int)));
  // copy host outputs to device outputs to give initial conditions
  *output = 0;
  checkCudaError(cudaMemcpy(dev_output, output, 1*sizeof(unsigned int),
                            cudaMemcpyHostToDevice));

  // calculate the number of blocks
  const unsigned int numberOfBlocks =
    min(maxNumberOfBlocks,
        unsigned(std::ceil(numberOfInputs / double(numberOfThreadsPerBlock))));

  *elapsedTime = DBL_MAX; // sigh, no numeric_limits

  // run the test repeatedly
  for (unsigned int trialNumber = 0;
       trialNumber < numberOfTrials; ++trialNumber) {

    // copy host outputs to device outputs to give initial conditions
    *output = 0;
    checkCudaError(cudaMemcpy(dev_output, output, 1*sizeof(unsigned int),
                              cudaMemcpyHostToDevice));

    // this forces the GPU to run another kernel, kind of like
    //  "resetting the cache" for the cpu versions.
    GpuUtilities::resetGpu();

    // Wait for any kernels to stop
    checkCudaError(cudaDeviceSynchronize());

    // start timing
    const TimeUtility::PreCpp11TimePoint tic = TimeUtility::getCurrentTime();

    if (cudaReductionStyle == SerialBlockReduction) {
      // TODO: you'll need to change this call to use shared memory
      doReduction_serialBlockReduction<<<numberOfBlocks,
        numberOfThreadsPerBlock>>>(dev_input,
                                   numberOfInputs,
                                   dev_output);
    } else if (cudaReductionStyle == Stupendous) {
      // TODO: you'll need to change this call to use shared memory
      doReduction_stupendous<<<numberOfBlocks,
        numberOfThreadsPerBlock>>>(dev_input,
                                   numberOfInputs,
                                   dev_output);
    } else {
      fprintf(stderr, "Unknown cudaReductionStyle\n");
      exit(1);
    }
    // see if there was an error in the kernel launch
    checkCudaError(cudaPeekAtLastError());

    // we really should do this here to be fair to thrust.
    // it reduces our performance with respect to the cpu, but we shouldn't
    //  be mean to thrust.
    checkCudaError(cudaMemcpy(output, dev_output,
                              1 * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));

    // Stop timing
    const TimeUtility::PreCpp11TimePoint toc = TimeUtility::getCurrentTime();
    const double thisTrialsElapsedTime =
      TimeUtility::getElapsedTime(tic, toc);
    *elapsedTime = std::min(*elapsedTime, thisTrialsElapsedTime);
  }

  // clean up
  checkCudaError(cudaFree(dev_input));
  checkCudaError(cudaFree(dev_output));

}

__global__
void
countThreads_kernel(unsigned int * totalCount) {
  atomicAdd(totalCount, 1);
}

void
runThrustTest(const unsigned int numberOfTrials,
              const unsigned int * const input,
              const unsigned int numberOfInputs,
              unsigned int * const output,
              double * const elapsedTime) {

  // allocate a junk data counter thing, similar to the one that is used
  //  to "reset the cache" for the cpu versions
  unsigned int *dev_junkDataCounter;
  checkCudaError(cudaMalloc((void **) &dev_junkDataCounter,
                            1*sizeof(unsigned int)));

  // allocate device-side input
  thrust::device_vector<unsigned int> dev_input(&input[0],
                                                &input[numberOfInputs]);

  *elapsedTime = DBL_MAX; // sigh, no numeric_limits

  // run the test repeatedly
  for (unsigned int trialNumber = 0;
       trialNumber < numberOfTrials; ++trialNumber) {

    // this forces the GPU to run another kernel, kind of like
    //  "resetting the cache" for the cpu versions.
    {
      const unsigned int resetGpuNumberOfThreadsPerBlock = 1024;
      const unsigned int resetGpuNumberOfBlocks = 1e8 /
        resetGpuNumberOfThreadsPerBlock;
      countThreads_kernel<<<resetGpuNumberOfBlocks,
        resetGpuNumberOfThreadsPerBlock>>>(dev_junkDataCounter);
      checkCudaError(cudaDeviceSynchronize());
    }
    // Normally I do this with resetGpu, but it's doing weird things here
    //  and I've been flailing on it for more than 2 hours and I want to go to
    //  sleep, so I give up.

    // start timing
    const TimeUtility::PreCpp11TimePoint tic = TimeUtility::getCurrentTime();

    // do the reduction
    *output = thrust::reduce(dev_input.begin(), dev_input.end());

    // Stop timing
    const TimeUtility::PreCpp11TimePoint toc = TimeUtility::getCurrentTime();
    const double thisTrialsElapsedTime =
      TimeUtility::getElapsedTime(tic, toc);
    *elapsedTime = std::min(*elapsedTime, thisTrialsElapsedTime);
  }

  // clean up
  checkCudaError(cudaFree(dev_junkDataCounter));

}
