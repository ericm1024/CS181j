// -*- C++ -*-
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <ctime>
#include <cfloat>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "ReductionRace_cuda.cuh"
#include "../GpuUtilities.h"

__device__
unsigned int
getLog2OfPowerOf2(unsigned int integerThatIsAPowerOf2) {
  unsigned int log2 = 0;
  while (integerThatIsAPowerOf2 >>= 1) {
    ++log2;
  }
  return log2;
}

__global__
void
doReduction_serialBlockReduction(const unsigned int * const input,
                                 const unsigned int numberOfInputs,
                                 unsigned int * const output) {
        extern __shared__ unsigned accumulators[];
        unsigned sum = 0;
        
        for (auto i = threadIdx.x + blockIdx.x * blockDim.x;
             i < numberOfInputs; i += blockDim.x * gridDim.x)
                sum += input[i];

        accumulators[threadIdx.x] = sum;
        __syncthreads();

        if (threadIdx.x == 0) {
                unsigned block_sum = 0;
                for (auto i = 0u; i < blockDim.x; ++i)
                        block_sum += accumulators[i];
                atomicAdd(output, block_sum);
        }
}

#if 0
__global__
void
doReduction_stupendous(const unsigned int * const input,
                       const unsigned int numberOfInputs,
                       const unsigned shmem_size,
                       unsigned int * const output) {

        // has shmem_size/sizeof(unsigned) elements
        extern __shared__ unsigned local_input[];
        __shared__ unsigned local_sum;

        auto inputs_per_block = ::min((unsigned)(shmem_size/sizeof(unsigned)),
                                      div_round_up(numberOfInputs, blockDim.x));

        // initialize block-local sum
        if (threadIdx.x == 0)
                local_sum = 0;
        __syncthreads();

        // each block reduces at most inputs_per_block elements at a time
        for (auto base = blockIdx.x * inputs_per_block;
             base < numberOfInputs;
             base += inputs_per_block * gridDim.x) {

                const auto end = ::min(base + inputs_per_block, numberOfInputs);
                const auto inputs_this_block = end - base;
                auto reduction_end = inputs_this_block;

                // **** nastyness incoming ****
                //
                // Here we implement a hybrid convergent tree reduction. It is 'hybrid' because
                // once the tree gets small enough (has fewer elements than we have threads
                // in this block), we do a butterfly reduction using only registers and __shfl_xor.
                // Thus we have need some nasty cases that determine if we load our block into
                // shared memory or not.
                // 
                // For large reductions (namely when inputs_this_block > blockDim.x), we want
                // to load the block of data we're going to reduce into shared memory first.
                // For small reductions, we don't need this step because we're never going to
                // store into stored memory, so the following pointer 'data' is where we
                // read from when we do the butterfly reduction.
                
                const unsigned *data = inputs_this_block > blockDim.x ? local_input : input + base;
                
                if (inputs_this_block > blockDim.x) {
                        // load the range that this block is going to reduce into shared memory
                        __syncthreads();
                        for (auto i = threadIdx.x; i < inputs_this_block; i += blockDim.x)
                                local_input[i] = input[base + i];
                        __syncthreads();

                        // reduce this block's range. Use shared memory until we have just a few values
                        // left...
                        
                        for (auto stride = div_round_up(reduction_end, 2u);
                             reduction_end > blockDim.x;
                             stride = div_round_up(reduction_end, 2u)) {

                                for (auto i = threadIdx.x; i + stride < reduction_end; i += blockDim.x)
                                        local_input[i] += local_input[i + stride];

                                reduction_end = stride;
                                __syncthreads();
                        }
                }

                // then use a butterfly reduction for the last few values. see this link:
                // http://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-examples-reduction
                auto sum = threadIdx.x < reduction_end ? data[threadIdx.x] : 0;
                for (auto i = warpSize/2; i>=1; i >>= 1)
                        sum += __shfl_xor(sum, i, 32);

                // accumulate warp local sum to block-local sum
                if (threadIdx.x % warpSize == 0)
                        atomicAdd(&local_sum, sum);
        }

        // once we've processed everything, accumulate block-local sum to global sum.
        __syncthreads();
        if (threadIdx.x == 0)
                atomicAdd(output, local_sum);
}
#else

__global__
void
doReduction_stupendous(const unsigned int * const input,
                       const unsigned int numberOfInputs,
                       const unsigned shmem_size,
                       unsigned int * const output) {

        auto sum = 0u;

        for (auto i = threadIdx.x + blockIdx.x * blockDim.x;
             i < numberOfInputs; i += blockDim.x * gridDim.x)
                sum += input[i];

        for (auto i = warpSize/2; i>=1; i >>= 1)
                sum += __shfl_xor(sum, i, 32);

        if (threadIdx.x % warpSize == 0)
                atomicAdd(output, sum);
}

#endif

void
runGpuTimingTest(const unsigned int numberOfTrials,
                 const unsigned int maxNumberOfBlocks,
                 const unsigned int numberOfThreadsPerBlock,
                 const CudaReductionStyle cudaReductionStyle,
                 const unsigned int * const input,
                 const unsigned int numberOfInputs,
                 unsigned int * const output,
                 double * const elapsedTime) {

        // allocate device-side input
        unsigned int *dev_input;
        checkCudaError(cudaMalloc((void **) &dev_input,
                                  numberOfInputs*sizeof(unsigned int)));
        checkCudaError(cudaMemcpy(dev_input, input,
                                  numberOfInputs*sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));

        // allocate device-side output
        unsigned int *dev_output;
        checkCudaError(cudaMalloc((void **) &dev_output, 1*sizeof(unsigned int)));
        // copy host outputs to device outputs to give initial conditions
        *output = 0;
        checkCudaError(cudaMemcpy(dev_output, output, 1*sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));

        // calculate the number of blocks
        const unsigned int numberOfBlocks =
                min(maxNumberOfBlocks,
                    unsigned(std::ceil(numberOfInputs / double(numberOfThreadsPerBlock))));

        *elapsedTime = DBL_MAX; // sigh, no numeric_limits

        // run the test repeatedly
        for (unsigned int trialNumber = 0;
             trialNumber < numberOfTrials; ++trialNumber) {

                // copy host outputs to device outputs to give initial conditions
                *output = 0;
                checkCudaError(cudaMemcpy(dev_output, output, 1*sizeof(unsigned int),
                                          cudaMemcpyHostToDevice));

                // this forces the GPU to run another kernel, kind of like
                //  "resetting the cache" for the cpu versions.
                GpuUtilities::resetGpu();

                // Wait for any kernels to stop
                checkCudaError(cudaDeviceSynchronize());

                // start timing
                const TimeUtility::PreCpp11TimePoint tic = TimeUtility::getCurrentTime();

                if (cudaReductionStyle == SerialBlockReduction) {
                        // TODO: you'll need to change this call to use shared memory
                        doReduction_serialBlockReduction<<<numberOfBlocks,
                                numberOfThreadsPerBlock,
                                numberOfThreadsPerBlock*sizeof(unsigned)
                                                        >>>(dev_input,
                                                            numberOfInputs,
                                                            dev_output);
                } else if (cudaReductionStyle == Stupendous) {
                        // TODO: you'll need to change this call to use shared memory
                        const auto shmem_size = 48*1000u;
                        doReduction_stupendous<<<numberOfBlocks,
                                numberOfThreadsPerBlock,
                                shmem_size
                                              >>>(dev_input,
                                                  numberOfInputs,
                                                  shmem_size,
                                                  dev_output);
                } else {
                        fprintf(stderr, "Unknown cudaReductionStyle\n");
                        exit(1);
                }
                // see if there was an error in the kernel launch
                checkCudaError(cudaPeekAtLastError());

                // we really should do this here to be fair to thrust.
                // it reduces our performance with respect to the cpu, but we shouldn't
                //  be mean to thrust.
                checkCudaError(cudaMemcpy(output, dev_output,
                                          1 * sizeof(unsigned int),
                                          cudaMemcpyDeviceToHost));

                // Stop timing
                const TimeUtility::PreCpp11TimePoint toc = TimeUtility::getCurrentTime();
                const double thisTrialsElapsedTime =
                        TimeUtility::getElapsedTime(tic, toc);
                *elapsedTime = std::min(*elapsedTime, thisTrialsElapsedTime);
        }

        // clean up
        checkCudaError(cudaFree(dev_input));
        checkCudaError(cudaFree(dev_output));

}

__global__
void
countThreads_kernel(unsigned int * totalCount) {
  atomicAdd(totalCount, 1);
}

void
runThrustTest(const unsigned int numberOfTrials,
              const unsigned int * const input,
              const unsigned int numberOfInputs,
              unsigned int * const output,
              double * const elapsedTime) {

  // allocate a junk data counter thing, similar to the one that is used
  //  to "reset the cache" for the cpu versions
  unsigned int *dev_junkDataCounter;
  checkCudaError(cudaMalloc((void **) &dev_junkDataCounter,
                            1*sizeof(unsigned int)));

  // allocate device-side input
  thrust::device_vector<unsigned int> dev_input(&input[0],
                                                &input[numberOfInputs]);

  *elapsedTime = DBL_MAX; // sigh, no numeric_limits

  // run the test repeatedly
  for (unsigned int trialNumber = 0;
       trialNumber < numberOfTrials; ++trialNumber) {

    // this forces the GPU to run another kernel, kind of like
    //  "resetting the cache" for the cpu versions.
    {
      const unsigned int resetGpuNumberOfThreadsPerBlock = 1024;
      const unsigned int resetGpuNumberOfBlocks = 1e8 /
        resetGpuNumberOfThreadsPerBlock;
      countThreads_kernel<<<resetGpuNumberOfBlocks,
        resetGpuNumberOfThreadsPerBlock>>>(dev_junkDataCounter);
      checkCudaError(cudaDeviceSynchronize());
    }
    // Normally I do this with resetGpu, but it's doing weird things here
    //  and I've been flailing on it for more than 2 hours and I want to go to
    //  sleep, so I give up.

    // start timing
    const TimeUtility::PreCpp11TimePoint tic = TimeUtility::getCurrentTime();

    // do the reduction
    *output = thrust::reduce(dev_input.begin(), dev_input.end());

    // Stop timing
    const TimeUtility::PreCpp11TimePoint toc = TimeUtility::getCurrentTime();
    const double thisTrialsElapsedTime =
      TimeUtility::getElapsedTime(tic, toc);
    *elapsedTime = std::min(*elapsedTime, thisTrialsElapsedTime);
  }

  // clean up
  checkCudaError(cudaFree(dev_junkDataCounter));

}
