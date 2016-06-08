// -*- C++ -*-
#ifndef MAIN4_CUDA_CUH
#define MAIN4_CUDA_CUH

void runGpuTimingTest(const unsigned int numberOfTrials,
                      const std::vector<float>& input,
                      const std::vector<float>& coefficients,
                      const unsigned int maxNumberOfBlocks,
                      const unsigned int numberOfThreadsPerBlock,
                      std::vector<float>* output,
                      double * elapsedTime);

#endif // MAIN4_CUDA_CUH
