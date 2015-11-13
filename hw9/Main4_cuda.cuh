// -*- C++ -*-
#ifndef MAIN4_CUDA_CUH
#define MAIN4_CUDA_CUH

void
runGpuTimingTest(const unsigned int numberOfRepeats,
                 const unsigned int maxNumberOfBlocks,
                 const unsigned int numberOfThreadsPerBlock,
                 double * elapsedTime);

#endif // MAIN4_CUDA_CUH
