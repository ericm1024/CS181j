// -*- C++ -*-
#include <cstdio>
#include <cfloat>

#include <cuda_runtime.h>
// These come from the cublas matrix multiplication example
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch"
#include <cublas_v2.h>
#pragma GCC diagnostic pop

#include "MatrixMultiplication_cuda.cuh"
#include "../GpuUtilities.h"

__global__
void
cudaDoNaiveMatrixMultiplication_kernel_rowTimesCol(const unsigned int matrixSize,
                                                   const double * leftMatrix,
                                                   const double * rightMatrix,
                                                   double * resultMatrix) {
  const unsigned int numberOfEntries = matrixSize * matrixSize;
  unsigned int elementIndex = blockIdx.x * blockDim.x + threadIdx.x;
  while (elementIndex < numberOfEntries) {
    const unsigned int row = elementIndex / matrixSize;
    const unsigned int col = elementIndex - row * matrixSize;
    double sum = 0;
    for (unsigned int dummy = 0; dummy < matrixSize; ++dummy) {
      sum +=
        leftMatrix[row * matrixSize + dummy] *
        rightMatrix[dummy + col * matrixSize];
    }
    resultMatrix[row * matrixSize + col] = sum;
    elementIndex += blockDim.x * gridDim.x;
  }
}

__global__
void
cudaDoNaiveMatrixMultiplication_kernel_rowTimesRow(const unsigned int matrixSize,
                                                   const double * leftMatrix,
                                                   const double * rightMatrix,
                                                   double * resultMatrix) {
  const unsigned int numberOfEntries = matrixSize * matrixSize;
  unsigned int elementIndex = blockIdx.x * blockDim.x + threadIdx.x;
  while (elementIndex < numberOfEntries) {
    const unsigned int row = elementIndex / matrixSize;
    const unsigned int col = elementIndex - row * matrixSize;
    double sum = 0;
    for (unsigned int dummy = 0; dummy < matrixSize; ++dummy) {
      sum +=
        leftMatrix[row * matrixSize + dummy] *
        rightMatrix[dummy * matrixSize + col];
    }
    resultMatrix[row * matrixSize + col] = sum;
    elementIndex += blockDim.x * gridDim.x;
  }
}

__global__
void
cudaDoRowMajorStorage_TiledMatrixMultiplication_kernel_global(const unsigned int matrixSize,
                                                              const unsigned int tileSize,
                                                              const double * leftMatrix,
                                                              const double * rightMatrix,
                                                              double * resultMatrix) {
  // assumption: tileSize is 16 or 32 and we have tileSize*tileSize threads per
  //  block, so we have 1 block per result tile
  if ((tileSize == 16 && blockDim.x == 256) == false &&
      (tileSize == 32 && blockDim.x == 1024) == false) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      printf("tiled versions assume either tile/block sizes of "
             "16/256 or 32/1024\n");
    }
    return;
  }

  // TODO: you can paste in your own code from last time or you can use this.

  const unsigned int numberOfTilesPerSide = matrixSize / tileSize;
  const unsigned int numberOfTiles = numberOfTilesPerSide * numberOfTilesPerSide;

  const unsigned int subRow = threadIdx.x / tileSize;
  const unsigned int subCol = threadIdx.x  - subRow * tileSize;

  unsigned int resultTileIndex = blockIdx.x;

  while (resultTileIndex < numberOfTiles) {

    // calculate result tile indices
    const unsigned int resultTileRow = resultTileIndex / numberOfTilesPerSide;
    const unsigned int resultTileCol = resultTileIndex  -
      resultTileRow * numberOfTilesPerSide;
    // calculate this entry's row and col
    const unsigned int row = resultTileRow * tileSize + subRow;
    const unsigned int col = resultTileCol * tileSize + subCol;
    // calculate the resultIndex
    const unsigned int resultIndex = row * matrixSize + col;

    double sum = 0;
    // for tileNumber in 0...numberOfTilesPerSide
    for (unsigned int tileNumber = 0;
         tileNumber < numberOfTilesPerSide; ++tileNumber) {

      const unsigned int leftBaseIndex =
        row * matrixSize + tileNumber * tileSize;
      const unsigned int rightBaseIndex =
        tileNumber * tileSize * matrixSize + col;

      for (unsigned int dummy = 0; dummy < tileSize; ++dummy) {
        sum +=
          leftMatrix[leftBaseIndex + dummy] *
          rightMatrix[rightBaseIndex + dummy * matrixSize];
      }
    }
    resultMatrix[resultIndex] = sum;
    resultTileIndex += gridDim.x;
  }
}

__global__
void
cudaDoRowMajorStorage_TiledMatrixMultiplication_kernel_shared(const unsigned int matrixSize,
                                                              const unsigned int tileSize,
                                                              const double * leftMatrix,
                                                              const double * rightMatrix,
                                                              double * resultMatrix) {
  // assumption: tileSize is 16 or 32 and we have tileSize*tileSize threads per
  //  block, so we have 1 block per result tile

  if (tileSize * tileSize != blockDim.x) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      printf("cannot do tiled matrix multiplication with tilesize %u and "
             "%u threads per block\n", tileSize, blockDim.x);
    }
    return;
  }

  // TODO: you can do it!

}

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
                 double * elapsedTime) {

  const unsigned int numberOfEntries = matrixSize * matrixSize;

  // copy leftMatrix to the gpu
  double * dev_leftMatrix;
  checkCudaError(cudaMalloc((void **) &dev_leftMatrix,
                            numberOfEntries * sizeof(double)));
  checkCudaError(cudaMemcpy(dev_leftMatrix, leftMatrix,
                            numberOfEntries * sizeof(double),
                            cudaMemcpyHostToDevice));

  // copy rightMatrix to the gpu
  double * dev_rightMatrix;
  checkCudaError(cudaMalloc((void **) &dev_rightMatrix,
                            numberOfEntries * sizeof(double)));
  checkCudaError(cudaMemcpy(dev_rightMatrix, rightMatrix,
                            numberOfEntries * sizeof(double),
                            cudaMemcpyHostToDevice));

  // allocate output matrix on the gpu
  double * dev_resultMatrix;
  checkCudaError(cudaMalloc((void **) &dev_resultMatrix,
                            numberOfEntries * sizeof(double)));
  checkCudaError(cudaMemcpy(dev_resultMatrix, resultMatrix,
                            numberOfEntries * sizeof(double),
                            cudaMemcpyHostToDevice));

  // calculate the number of blocks
  const unsigned int numberOfBlocks =
    min(maxNumberOfBlocks,
        unsigned(std::ceil(numberOfEntries / double(numberOfThreadsPerBlock))));

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

    if (cudaStyle == Naive_RowMajorTimesColMajor) {
      // run the kernel
      cudaDoNaiveMatrixMultiplication_kernel_rowTimesCol<<<numberOfBlocks,
        numberOfThreadsPerBlock>>>(matrixSize,
                                   dev_leftMatrix,
                                   dev_rightMatrix,
                                   dev_resultMatrix);
    } else if (cudaStyle == Naive_RowMajorTimesRowMajor) {
      // run the kernel
      cudaDoNaiveMatrixMultiplication_kernel_rowTimesRow<<<numberOfBlocks,
        numberOfThreadsPerBlock>>>(matrixSize,
                                   dev_leftMatrix,
                                   dev_rightMatrix,
                                   dev_resultMatrix);
    } else if (cudaStyle == RowMajorStorage_TiledMultiplication_Global) {
      cudaDoRowMajorStorage_TiledMatrixMultiplication_kernel_global<<<numberOfBlocks,
        numberOfThreadsPerBlock>>>(matrixSize,
                                   tileSize,
                                   dev_leftMatrix,
                                   dev_rightMatrix,
                                   dev_resultMatrix);
    } else if (cudaStyle == RowMajorStorage_TiledMultiplication_Shared) {
      cudaDoRowMajorStorage_TiledMatrixMultiplication_kernel_shared<<<numberOfBlocks,
        numberOfThreadsPerBlock,
        2 * tileSize * tileSize * sizeof(double)>>>(matrixSize,
                                                    tileSize,
                                                    dev_leftMatrix,
                                                    dev_rightMatrix,
                                                    dev_resultMatrix);
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

  // copy over the output
  checkCudaError(cudaMemcpy(resultMatrix, dev_resultMatrix,
                            numberOfEntries * sizeof(double),
                            cudaMemcpyDeviceToHost));

  // clean up
  checkCudaError(cudaFree(dev_leftMatrix));
  checkCudaError(cudaFree(dev_rightMatrix));
  checkCudaError(cudaFree(dev_resultMatrix));
}

void
multiplyRowMajorMatricesUsingCublas(const unsigned int numberOfTrials,
                                    const unsigned int matrixSize,
                                    const double * leftMatrix,
                                    const double * rightMatrix,
                                    double * resultMatrix,
                                    double * elapsedTime) {

  const unsigned int numberOfEntries = matrixSize * matrixSize;

  // allocate device memory
  double * dev_leftMatrix;
  double * dev_rightMatrix;
  double * dev_resultMatrix;
  checkCudaError(cudaMalloc((void **) &dev_leftMatrix,
                            numberOfEntries * sizeof(double)));
  checkCudaError(cudaMalloc((void **) &dev_rightMatrix,
                            numberOfEntries * sizeof(double)));
  checkCudaError(cudaMalloc((void **) &dev_resultMatrix,
                            numberOfEntries * sizeof(double)));
  // copy matrices to the device
  checkCudaError(cudaMemcpy(dev_leftMatrix, leftMatrix,
                            numberOfEntries * sizeof(double),
                            cudaMemcpyHostToDevice));
  checkCudaError(cudaMemcpy(dev_rightMatrix, rightMatrix,
                            numberOfEntries * sizeof(double),
                            cudaMemcpyHostToDevice));

  const double alpha = 1.0f;
  const double beta  = 0.0f;
  cublasHandle_t handle;

  cublasCreate(&handle);

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

    // perform the multiply
    cublasDgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N, // don't transpose
                matrixSize, matrixSize, matrixSize, // sizes
                &alpha, // no scalar premultiply
                dev_rightMatrix, matrixSize, // left matrix
                dev_leftMatrix, matrixSize, // right matrix
                &beta, // don't premultiply result by anything
                dev_resultMatrix, matrixSize);

    // wait for multiplication to finish
    checkCudaError(cudaDeviceSynchronize());

    // Stop timing
    const TimeUtility::PreCpp11TimePoint toc = TimeUtility::getCurrentTime();
    const double thisTrialsElapsedTime =
      TimeUtility::getElapsedTime(tic, toc);
    *elapsedTime = std::min(*elapsedTime, thisTrialsElapsedTime);
  }

  // copy result from device to host
  cudaMemcpy(resultMatrix, dev_resultMatrix,
             numberOfEntries * sizeof(double),
             cudaMemcpyDeviceToHost);

  // Destroy the handle
  cublasDestroy(handle);

  // clean up memory
  checkCudaError(cudaFree(dev_leftMatrix));
  checkCudaError(cudaFree(dev_rightMatrix));
  checkCudaError(cudaFree(dev_resultMatrix));
}
