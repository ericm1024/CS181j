// -*- C++ -*-
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <cfloat>

#include <cuda_runtime.h>

#include "KMeansClustering_cuda.cuh"
#include "../GpuUtilities.h"

template <class T>
__global__
void
setArray_kernel(const unsigned int arraySize,
                const T value,
                T * __restrict__ array) {

  unsigned int arrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
  while (arrayIndex < arraySize) {
    array[arrayIndex] = value;
    arrayIndex += blockDim.x * gridDim.x;
  }

}

void
runGpuTimingTest(const unsigned int numberOfTrials,
                 const unsigned int maxNumberOfBlocks,
                 const unsigned int numberOfThreadsPerBlock,
                 const float * const points_Cpu_AoS,
                 const unsigned int numberOfPoints,
                 const float * const startingCentroids_Cpu_AoS,
                 const unsigned int numberOfCentroids,
                 const unsigned int numberOfIterations,
                 float * const finalCentroids_Cpu_AoS,
                 float * const elapsedTime) {

  // Make Cpu versions of the data that we'll need to ship to the Gpu
  float * const points_gpuLayout = new float[numberOfPoints * 3];
  for (unsigned int pointIndex = 0;
       pointIndex < numberOfPoints; ++pointIndex) {
    for (unsigned int coordinate = 0; coordinate < 3; ++coordinate) {
      points_gpuLayout[coordinate * numberOfPoints + pointIndex] =
        points_Cpu_AoS[pointIndex * 3 + coordinate];
    }
  }

  float * const startingCentroids_gpuLayout = new float[numberOfCentroids * 3];
  for (unsigned int centroidIndex = 0;
       centroidIndex < numberOfCentroids; ++centroidIndex) {
    for (unsigned int coordinate = 0; coordinate < 3; ++coordinate) {
      startingCentroids_gpuLayout[coordinate * numberOfCentroids + centroidIndex] =
        startingCentroids_Cpu_AoS[centroidIndex * 3 + coordinate];
    }
  }

  // Allocate device-side points
  float * dev_points;
  checkCudaError(cudaMalloc((void **) &dev_points,
                            numberOfPoints * 3 * sizeof(float)));
  float * dev_centroids;
  checkCudaError(cudaMalloc((void **) &dev_centroids,
                            numberOfCentroids * 3 * sizeof(float)));
  float * dev_nextCentroids;
  checkCudaError(cudaMalloc((void **) &dev_nextCentroids,
                            numberOfCentroids * 3 * sizeof(float)));
  unsigned int * dev_nextCentroidCounts;
  checkCudaError(cudaMalloc((void **) &dev_nextCentroidCounts,
                            numberOfCentroids * 1 * sizeof(unsigned int)));

  // Copy host inputs to device
  checkCudaError(cudaMemcpy(dev_points, points_gpuLayout,
                            numberOfPoints * 3 * sizeof(float),
                            cudaMemcpyHostToDevice));

  // set next centroids and next centroid counts to zero
  setArray_kernel<unsigned int><<<ceil(numberOfCentroids / 512.),
    512>>>(numberOfCentroids,
           0,
           dev_nextCentroidCounts);
  // set next centroids and next centroid counts to zero
  setArray_kernel<float><<<ceil(numberOfCentroids * 3 / 512.),
    512>>>(numberOfCentroids * 3,
           0,
           dev_nextCentroids);

  // calculate the number of blocks
  const unsigned int numberOfBlocksForPoints =
    min(maxNumberOfBlocks,
        (unsigned int)ceil(numberOfPoints/float(numberOfThreadsPerBlock)));
  // all this work, and it's honestly just like 2
  const unsigned int numberOfBlocksForCentroids =
    min(maxNumberOfBlocks,
        (unsigned int)ceil(numberOfCentroids/float(numberOfThreadsPerBlock)));

  *elapsedTime = DBL_MAX; // sigh, no numeric_limits

  // run the test repeatedly
  for (unsigned int trialNumber = 0;
       trialNumber < numberOfTrials; ++trialNumber) {

    // Reset intermediate values for next calculation
    checkCudaError(cudaMemcpy(dev_centroids, startingCentroids_gpuLayout,
                              numberOfCentroids * 3 * sizeof(float),
                              cudaMemcpyHostToDevice));

    // this forces the GPU to run another kernel, kind of like
    //  "resetting the cache" for the cpu versions.
    GpuUtilities::resetGpu();

    // Wait for any kernels to stop
    checkCudaError(cudaDeviceSynchronize());

    // Start timing
    const TimeUtility::PreCpp11TimePoint tic = TimeUtility::getCurrentTime();


    // For each of a fixed number of iterations
    for (unsigned int iterationNumber = 0;
         iterationNumber < numberOfIterations; ++iterationNumber) {

      // TODO: one or more kernels

      // See if there was an error in the kernel launch
      checkCudaError(cudaPeekAtLastError());
    }

    // Wait for the kernels to stop
    checkCudaError(cudaDeviceSynchronize());

    // Stop timing
    const TimeUtility::PreCpp11TimePoint toc = TimeUtility::getCurrentTime();
    const float thisTrialsElapsedTime =
      TimeUtility::getElapsedTime(tic, toc);
    *elapsedTime = std::min(*elapsedTime, thisTrialsElapsedTime);
  }

  // copy device outputs back to host
  float * finalCentroids_gpuLayout = new float[numberOfCentroids * 3];
  checkCudaError(cudaMemcpy(finalCentroids_gpuLayout, dev_centroids,
                            numberOfCentroids * 3 * sizeof(float),
                            cudaMemcpyDeviceToHost));

  for (unsigned int centroidIndex = 0;
       centroidIndex < numberOfCentroids; ++centroidIndex) {
    for (unsigned int coordinate = 0; coordinate < 3; ++coordinate) {
      finalCentroids_Cpu_AoS[centroidIndex * 3 + coordinate] =
        finalCentroids_gpuLayout[coordinate * numberOfCentroids + centroidIndex];
    }
  }

  checkCudaError(cudaFree(dev_points));
  checkCudaError(cudaFree(dev_centroids));
  checkCudaError(cudaFree(dev_nextCentroids));
  checkCudaError(cudaFree(dev_nextCentroidCounts));
  delete[] points_gpuLayout;
  delete[] startingCentroids_gpuLayout;
  delete[] finalCentroids_gpuLayout;
}
