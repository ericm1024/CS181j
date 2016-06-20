// -*- C++ -*-
#include <cstdio>
#include <cfloat>
#include <iostream>

#include <cuda_runtime.h>
// These come from the cublas matrix multiplication example
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch"
#include <cublas_v2.h>
#pragma GCC diagnostic pop

#include "MatrixMultiplication_cuda.cuh"
#include "../GpuUtilities.h"

//#define CUDA_DEBUG

__global__
void
cudaDoNaiveMatrixMultiplication_kernel_rowTimesCol(const unsigned int matrixSize,
                                                   const double * leftMatrix,
                                                   const double * rightMatrix,
                                                   double * resultMatrix) {
        for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
             i < matrixSize*matrixSize; i += blockDim.x * gridDim.x) {
                double accum = 0;
                auto row = i/matrixSize;
                auto col = i%matrixSize;

                for (unsigned j = 0; j < matrixSize; ++j)
                        accum += leftMatrix[row*matrixSize + j]
                                * rightMatrix[col*matrixSize + j];

                resultMatrix[row*matrixSize + col] = accum;
        }
}

__global__
void
cudaDoNaiveMatrixMultiplication_kernel_rowTimesRow(const unsigned int matrixSize,
                                                   const double * leftMatrix,
                                                   const double * rightMatrix,
                                                   double * resultMatrix) {
        for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
             i < matrixSize*matrixSize; i += blockDim.x * gridDim.x) {
                double accum = 0;
                auto row = i/matrixSize;
                auto col = i%matrixSize;
                
                for (unsigned j = 0; j < matrixSize; ++j)
                        accum += leftMatrix[row*matrixSize + j]
                                * rightMatrix[j*matrixSize + col];

                resultMatrix[row*matrixSize + col] = accum;
        }
}

__global__
void
cudaDoRowMajorStorage_TiledMatrixMultiplication_kernel_global(const unsigned int matrixSize,
                                                              const unsigned int tileSize,
                                                              const double * leftMatrix,
                                                              const double * rightMatrix,
                                                              double * resultMatrix) {

        extern __shared__ uint8_t mem[];
        
        double *left_tile = (double*)mem; // size = tileSize*tileSize doubles
        double *right_tile = left_tile + tileSize*tileSize;

        const auto tiles_per_side = matrixSize/tileSize;
        
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

        // each block handles one of the resulting tiles
        for (auto i = blockIdx.x; i < tiles_per_side*tiles_per_side; i += gridDim.x) {
                const auto base_row = tileSize*(i/tiles_per_side);
                const auto base_col = tileSize*(i%tiles_per_side);

                const auto thread_row = threadIdx.x/tileSize;
                const auto thread_col = threadIdx.x%tileSize;

                double accum = 0;
                
                // for each pair of tiles
                for (auto j = 0u; j < matrixSize; j += tileSize) {
                        // load the current tiles into memory;
                        __syncthreads();
                        left_tile[threadIdx.x] = leftMatrix[(base_row + thread_row) * matrixSize
                                                            + thread_col + j];
                        right_tile[threadIdx.x] = rightMatrix[(thread_row + j) * matrixSize
                                                              + base_col + thread_col];
                        __syncthreads();

                        for (auto k = 0u; k < tileSize; ++k)
                                accum += left_tile[thread_row*tileSize + k]
                                        * right_tile[k*tileSize + thread_col];
                }

                resultMatrix[(base_row + thread_row) * matrixSize
                             + base_col + thread_col] = accum;
        }
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
                                numberOfThreadsPerBlock,
                                tileSize*tileSize*2*sizeof(double)>>>(matrixSize,
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
