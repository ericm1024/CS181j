// -*- C++ -*-
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <ctime>
#include <algorithm>
#include <memory>

#include <cuda_runtime.h>

#include "../GpuUtilities.h"
#include "../dev_mem.hpp"

#include "Main3_cuda.cuh"

// sin(cos(exp(tan(x))));
__global__
static void kernel(const double * __restrict__ input,
                   double * __restrict__ output,
                   const unsigned size)
{
        for (unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < size;
             idx += blockDim.x * gridDim.x)
                output[idx] = sin(cos(exp(tan(input[idx]))));
}

// TODO: you'll probably need to change this function call
void
calculateWeirdFunction_cuda(const double *input,
                            double *output,
                            const unsigned size,
                            const unsigned maxNumberOfBlocks,
                            const unsigned numberOfThreadsPerBlock)
{       
        dev_mem<const double> dev_in{input, size};
        dev_mem<double> dev_out{size};

        // calculate the number of blocks
        unsigned nr_blocks = std::min(maxNumberOfBlocks,
                                      (unsigned)std::ceil(size/double(numberOfThreadsPerBlock)));

        // run kernel
        kernel<<<nr_blocks, numberOfThreadsPerBlock>>>(dev_in, dev_out, size);

        dev_out.write_to(output, size);
}
