// -*- C++ -*-
#ifndef KMEANSCLUSTERING_CUDA_CUH
#define KMEANSCLUSTERING_CUDA_CUH

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
                 float * elapsedTime);

#endif // KMEANSCLUSTERING_CUDA_CUH
