// -*- C++ -*-
#ifndef BINARY_SEARCH_CUDA_CUH
#define BINARY_SEARCH_CUDA_CUH

void
runGpuTimingTest(const unsigned int numberOfTrials,
                 const unsigned int maxNumberOfBlocks,
                 const unsigned int numberOfThreadsPerBlock,
                 const unsigned int * sortedNumbers,
                 const unsigned int numberOfSortedNumbers,
                 const unsigned int * input,
                 const unsigned int inputSize,
                 bool * output,
                 double * elapsedTime);

#endif // BINARY_SEARCH_CUDA_CUH
