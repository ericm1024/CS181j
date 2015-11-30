// -*- C++ -*-
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <ctime>
#include <cfloat>

#include <cuda_runtime.h>

#include "ManyMatrixMultiplications_cuda.cuh"
#include "../GpuUtilities.h"

__global__
void
kernel_nextThreadNextEntry_serialMatrices(const unsigned long numberOfMatricesToMultiply,
                                          const unsigned long matrixSize,
                                          const double * const __restrict__ dev_leftMatrices,
                                          const double * const __restrict__ dev_rightMatrices,
                                          double * const __restrict__ dev_resultMatrices) {

  // TODO:

}

__global__
void
kernel_nextThreadNextMatrix_deepEntries(const unsigned long numberOfMatricesToMultiply,
                                        const unsigned long matrixSize,
                                        const double * const __restrict__ dev_leftMatrices,
                                        const double * const __restrict__ dev_rightMatrices,
                                        double * const __restrict__ dev_resultMatrices) {

  // TODO:

}

void
runGpuTimingTest(const unsigned int numberOfTrials,
                 const unsigned int maxNumberOfBlocks,
                 const unsigned int numberOfThreadsPerBlock,
                 const CudaManyMatrixMultiplicationStyle cudaManyMatrixMultiplicationStyle,
                 const unsigned int numberOfMatricesToMultiply,
                 const unsigned int matrixSize,
                 const double * leftMatrices,
                 const double * rightMatrices,
                 double * resultMatrices,
                 double * elapsedTime) {

  const unsigned int numberOfEntriesInAllMatrices =
    numberOfMatricesToMultiply * matrixSize * matrixSize;

  // allocate device-side matrices
  double * dev_leftMatrices;
  checkCudaError(cudaMalloc((void **) &dev_leftMatrices,
                            numberOfEntriesInAllMatrices*sizeof(double)));
  checkCudaError(cudaMemcpy(dev_leftMatrices, leftMatrices,
                            numberOfEntriesInAllMatrices*sizeof(double),
                            cudaMemcpyHostToDevice));
  double * dev_rightMatrices;
  checkCudaError(cudaMalloc((void **) &dev_rightMatrices,
                            numberOfEntriesInAllMatrices*sizeof(double)));
  checkCudaError(cudaMemcpy(dev_rightMatrices, rightMatrices,
                            numberOfEntriesInAllMatrices*sizeof(double),
                            cudaMemcpyHostToDevice));

  // allocate device-side result matrices
  double * dev_resultMatrices;
  checkCudaError(cudaMalloc((void **) &dev_resultMatrices,
                            numberOfEntriesInAllMatrices*sizeof(double)));

  // calculate the number of blocks
  const unsigned int numberOfBlocks =
    min(maxNumberOfBlocks,
        (unsigned int)ceil(numberOfEntriesInAllMatrices /
                           double(numberOfThreadsPerBlock)));

  *elapsedTime = DBL_MAX; // sigh, no numeric_limits

  // run the test repeatedly
  for (unsigned int trialNumber = 0;
       trialNumber < numberOfTrials; ++trialNumber) {

    // this forces the GPU to run another kernel, kind of like
    //  "resetting the cache" for the cpu versions.
    GpuUtilities::resetGpu();

    // Wait for any kernels to stop
    checkCudaError(cudaDeviceSynchronize());

    // Start timing
    const TimeUtility::PreCpp11TimePoint tic = TimeUtility::getCurrentTime();

    // run the kernel
    if (cudaManyMatrixMultiplicationStyle ==
        NextThreadNextEntry_serialMatrices) {
      kernel_nextThreadNextEntry_serialMatrices<<<numberOfBlocks,
        numberOfThreadsPerBlock>>>(numberOfMatricesToMultiply,
                                   matrixSize,
                                   dev_leftMatrices,
                                   dev_rightMatrices,
                                   dev_resultMatrices);
    } else if (cudaManyMatrixMultiplicationStyle ==
               NextThreadNextMatrix_deepEntries) {
      kernel_nextThreadNextMatrix_deepEntries<<<numberOfBlocks,
        numberOfThreadsPerBlock>>>(numberOfMatricesToMultiply,
                                   matrixSize,
                                   dev_leftMatrices,
                                   dev_rightMatrices,
                                   dev_resultMatrices);
    }

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

  // copy back the output matrices
  checkCudaError(cudaMemcpy(resultMatrices, dev_resultMatrices,
                            numberOfEntriesInAllMatrices*sizeof(double),
                            cudaMemcpyDeviceToHost));

  // clean up
  checkCudaError(cudaFree(dev_leftMatrices));
  checkCudaError(cudaFree(dev_rightMatrices));
  checkCudaError(cudaFree(dev_resultMatrices));
}
