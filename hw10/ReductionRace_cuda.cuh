// -*- C++ -*-
#ifndef REDUCTION_RACE_CUDA_CUH
#define REDUCTION_RACE_CUDA_CUH

enum CudaReductionStyle {SerialBlockReduction,
                         Stupendous};

void
runGpuTimingTest(const unsigned int numberOfTrials,
                 const unsigned int maxNumberOfBlocks,
                 const unsigned int numberOfThreadsPerBlock,
                 const CudaReductionStyle cudaReductionStyle,
                 const unsigned int * input,
                 const unsigned int numberOfInputs,
                 unsigned int * output,
                 double * elapsedTime);

void
runThrustTest(const unsigned int numberOfTrials,
              const unsigned int * input,
              const unsigned int numberOfInputs,
              unsigned int * output,
              double * elapsedTime);

#endif // REDUCTION_RACE_CUDA_CUH
