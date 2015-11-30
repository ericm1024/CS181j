// -*- C++ -*-
#ifndef MANYMATRIXMULTIPLICATIONS_CUDA_CUH
#define MANYMATRIXMULTIPLICATIONS_CUDA_CUH

enum CudaManyMatrixMultiplicationStyle {NextThreadNextEntry_serialMatrices,
                                        NextThreadNextMatrix_deepEntries};

void
runGpuTimingTest(const unsigned int numberOfTrials,
                 const unsigned int maxNumberOfBlocks,
                 const unsigned int numberOfThreadsPerBlock,
                 const CudaManyMatrixMultiplicationStyle cudaMatrixMultiplicationStyle,
                 const unsigned int numberOfMatricesToMultiply,
                 const unsigned int matrixSize,
                 const double * leftMatrices,
                 const double * rightMatrices,
                 double * resultMatrices,
                 double * elapsedTime);

#endif // MANYMATRIXMULTIPLICATIONS_CUDA_CUH
