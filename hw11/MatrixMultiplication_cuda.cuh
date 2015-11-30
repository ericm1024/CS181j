// -*- C++ -*-
#ifndef MATRIX_MULTIPLICATION_CUDA_CUH
#define MATRIX_MULTIPLICATION_CUDA_CUH

enum CudaMatrixMultiplicationStyle {Naive_RowMajorTimesColMajor,
                                    Naive_RowMajorTimesRowMajor,
                                    RowMajorStorage_TiledMultiplication_Global,
                                    RowMajorStorage_TiledMultiplication_Shared};

void
runGpuTimingTest(const unsigned int numberOfTrials,
                 const unsigned int maxNumberOfBlocks,
                 const unsigned int numberOfThreadsPerBlock,
                 const unsigned int matrixSize,
                 const unsigned int tileSize,
                 const CudaMatrixMultiplicationStyle cudaStyle,
                 const double * leftMatrix,
                 const double * rightMatrix,
                 double * resultMatrix,
                 double * elapsedTime);

void
multiplyRowMajorMatricesUsingCublas(const unsigned int numberOfTrials,
                                    const unsigned int matrixSize,
                                    const double * leftMatrix,
                                    const double * rightMatrix,
                                    double * resultMatrix,
                                    double * elapsedTime);

#endif // MATRIX_MULTIPLICATION_CUDA_CUH
