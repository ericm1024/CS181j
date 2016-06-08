// -*- C++ -*-
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <cfloat>

#include <iostream>

#include <cuda_runtime.h>

#include "BinarySearch_cuda.cuh"
#include "../GpuUtilities.h"

__global__
static void
findKeysInSortedNumbers_kernel(const unsigned int * __restrict__ sortedNumbers,
                               const unsigned int numberOfSortedNumbers,
                               const unsigned int * __restrict__ input,
                               const unsigned int inputSize,
                               bool * __restrict__ output) {

        for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
             i < inputSize; i += blockDim.x * gridDim.x) {

                unsigned first = 0, last = numberOfSortedNumbers - 1;
                auto key = input[i];

                output[i] = false;

                while (first <= last) {
                        unsigned midx = first + (last - first)/2;
                        auto mid = sortedNumbers[midx];
                        if (mid < key)
                                first = midx+1;
                        else if (mid > key)
                                last = midx-1;
                        else {
                                output[i] = true;
                                break;
                        }
                }
        }
}

void
runGpuTimingTest(const unsigned int numberOfTrials,
                 const unsigned int maxNumberOfBlocks,
                 const unsigned int numberOfThreadsPerBlock,
                 const unsigned int * sortedNumbers,
                 const unsigned int numberOfSortedNumbers,
                 const unsigned int * input,
                 const unsigned int inputSize,
                 bool * output,
                 double * elapsedTime) {

  // allocate device-side inputs
  unsigned int * dev_sortedNumbers;
  checkCudaError(cudaMalloc((void **) &dev_sortedNumbers,
                            numberOfSortedNumbers*sizeof(unsigned int)));
  unsigned int * dev_input;
  checkCudaError(cudaMalloc((void **) &dev_input,
                            inputSize*sizeof(unsigned int)));

  // copy host inputs to device
  checkCudaError(cudaMemcpy(dev_sortedNumbers, sortedNumbers,
                            numberOfSortedNumbers*sizeof(unsigned int),
                            cudaMemcpyHostToDevice));
  checkCudaError(cudaMemcpy(dev_input, input,
                            inputSize*sizeof(unsigned int),
                            cudaMemcpyHostToDevice));

  // allocate device-side outputs
  bool * dev_output;
  checkCudaError(cudaMalloc((void **) &dev_output,
                            inputSize*sizeof(bool)));

  // calculate the number of blocks
  const unsigned int numberOfBlocks =
    min(maxNumberOfBlocks,
        (unsigned int)ceil(inputSize/double(numberOfThreadsPerBlock)));

  *elapsedTime = DBL_MAX; // sigh, no numeric_limits

  // run the test repeatedly
  for (unsigned int trialNumber = 0;
       trialNumber < numberOfTrials; ++trialNumber) {

          std::cout << __func__ << "trialNumber=" << trialNumber
                    << " maxNumberOfBlocks=" << maxNumberOfBlocks
                    << " numberOfThreadsPerBlock=" << numberOfThreadsPerBlock
                    << " numberOfSortedNumbers=" << numberOfSortedNumbers
                    << std::endl;
          
    // this forces the GPU to run another kernel, kind of like
    //  "resetting the cache" for the cpu versions.
    GpuUtilities::resetGpu();

    // Wait for any kernels to stop
    checkCudaError(cudaDeviceSynchronize());

    // Start timing
    const TimeUtility::PreCpp11TimePoint tic = TimeUtility::getCurrentTime();

    // run kernel
    findKeysInSortedNumbers_kernel<<<numberOfBlocks,
      numberOfThreadsPerBlock>>>(dev_sortedNumbers,
                                 numberOfSortedNumbers,
                                 dev_input,
                                 inputSize,
                                 dev_output);
    // see if there was an error in the kernel launch
    checkCudaError(cudaPeekAtLastError());

    // wait for the kernel to stop
    checkCudaError(cudaDeviceSynchronize());

    // Stop timing
    const TimeUtility::PreCpp11TimePoint toc = TimeUtility::getCurrentTime();
    const double thisTrialsElapsedTime =
      TimeUtility::getElapsedTime(tic, toc);
    *elapsedTime = std::min(*elapsedTime, thisTrialsElapsedTime);
  }

  // copy device outputs back to host
  checkCudaError(cudaMemcpy(output, dev_output, inputSize*sizeof(bool),
                            cudaMemcpyDeviceToHost));

  checkCudaError(cudaFree(dev_sortedNumbers));
  checkCudaError(cudaFree(dev_input));
  checkCudaError(cudaFree(dev_output));
}
