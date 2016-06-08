// -*- C++ -*-
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <ctime>
#include <stdint.h>
#include <cfloat>

// All of the cuda functions are defined here.
#include <cuda_runtime.h>

// These utilities are used on the GPU assignments
#include "../GpuUtilities.h"


#define USE_SHARED_MEM

__global__
static void kernel(const float * input, float *output, unsigned size,
                   const float * coefficients, unsigned nr_coefficients)
{
#ifdef USE_SHARED_MEM
        extern __shared__ float local_coefficients[];

        for (unsigned idx = threadIdx.x; idx < nr_coefficients; idx += blockDim.x)
                local_coefficients[idx] = coefficients[idx];

        __syncthreads();
#endif 
        
        for (unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < size;
             idx += blockDim.x * gridDim.x) {
                float in = input[idx];
                float term = 1;
                float out = 0;

                for (unsigned j = 0; j < nr_coefficients; ++j) {
#ifdef USE_SHARED_MEM
                        out += term * local_coefficients[j];
#else 
                        out += term * coefficients[j];
#endif
                        term *= in;
                }

                output[idx] = out;
        }
}

void
runGpuTimingTest(const unsigned int numberOfTrials,
                 const std::vector<float>& input,
                 const std::vector<float>& coefficients,
                 const unsigned int maxNumberOfBlocks,
                 const unsigned int numberOfThreadsPerBlock,
                 std::vector<float>* output,
                 double * elapsedTime) {

        auto dev_in = make_dev_mem(input.data(), input.size());
        auto dev_coefficients = make_dev_mem(coefficients.data(),
                                             coefficients.size());
        dev_mem<float> dev_out{(*output).size()};

        const unsigned nr_blocks = std::min(maxNumberOfBlocks,
                                            (unsigned)std::ceil(input.size()/double(numberOfThreadsPerBlock)));
        

        *elapsedTime = std::numeric_limits<double>::max();

        // run the test repeatedly
        for (unsigned int trialNumber = 0;
             trialNumber < numberOfTrials; ++trialNumber) {

                // this forces the GPU to run another kernel, kind of like
                //  "resetting the cache" for the cpu versions.
                GpuUtilities::resetGpu();

                // start timing
                const auto tic = TimeUtility::getCurrentTime();

                // TODO: run kernel
                kernel<<<nr_blocks,
                        numberOfThreadsPerBlock,
                        dev_coefficients.size() * sizeof(float)
                      >>>
                        (dev_in, dev_out, dev_in.size(),
                         dev_coefficients, dev_coefficients.size());

                // see if there was an error in the kernel launch
                checkCudaError(cudaPeekAtLastError());

                // wait for the kernel to stop
                checkCudaError(cudaDeviceSynchronize());

                // stop timing
                const auto toc = TimeUtility::getCurrentTime();
                const double thisTrialsElapsedTime =
                        TimeUtility::getElapsedTime(tic, toc);
                *elapsedTime = std::min(*elapsedTime, thisTrialsElapsedTime);
        }

        dev_out.write_to((*output).data(), (*output).size());
}
