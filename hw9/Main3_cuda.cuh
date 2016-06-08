// -*- C++ -*-
#ifndef MAIN3_CUDA_CUH
#define MAIN3_CUDA_CUH

// TODO: you'll probably need to change this function call
void
calculateWeirdFunction_cuda(const double *input,
                            double *output,
                            const unsigned size,
                            const unsigned int maxNumberOfBlocks,
                            const unsigned int numberOfThreadsPerBlock);

#endif // MAIN3_CUDA_CUH
