// -*- C++ -*-
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <ctime>

#include <cuda_runtime.h>

#include "../GpuUtilities.h"

#include "Main0_cuda.cuh"

__global__
void
sayHello_kernel(const double numberToPassAsFunctionArgument,
                const double * const __restrict__ numberToPassThroughDeviceMemory,
                double * productOfTheTwoNumbers) {

  // get the number
  const double numberFromDeviceMemory =
    *numberToPassThroughDeviceMemory;

  // compute the product
  *productOfTheTwoNumbers =
    numberToPassAsFunctionArgument * numberFromDeviceMemory;

  // say hello
  printf("Hi from thread %4u of %4u in this block %5u of %5u\n",
         threadIdx.x, blockDim.x, blockIdx.x, gridDim.x);
}

void
sayHello_cuda(const double numberToPassAsFunctionArgument,
              const double * const numberToPassThroughDeviceMemory,
              double * productOfTheTwoNumbers) {

  // allocate device-side inputs
  double * dev_numberToPassThroughDeviceMemory;
  checkCudaError(cudaMalloc((void **) &dev_numberToPassThroughDeviceMemory,
                            1*sizeof(double)));

  // copy host inputs to device
  checkCudaError(cudaMemcpy(dev_numberToPassThroughDeviceMemory,
                            numberToPassThroughDeviceMemory,
                            1*sizeof(double),
                            cudaMemcpyHostToDevice));

  // allocate device-side outputs
  double * dev_productOfTheTwoNumbers;
  checkCudaError(cudaMalloc((void **) &dev_productOfTheTwoNumbers,
                            1*sizeof(double)));

  // run kernel
  const unsigned int numberOfBlocks = 1;
  const unsigned int numberOfThreadsPerBlock = 1;
  sayHello_kernel<<<numberOfBlocks,
    numberOfThreadsPerBlock>>>(numberToPassAsFunctionArgument,
                               dev_numberToPassThroughDeviceMemory,
                               dev_productOfTheTwoNumbers);
  // see if there was an error in the kernel launch
  checkCudaError(cudaPeekAtLastError());

  // copy device outputs back to host
  checkCudaError(cudaMemcpy(productOfTheTwoNumbers, dev_productOfTheTwoNumbers,
                            1*sizeof(double),
                            cudaMemcpyDeviceToHost));

  checkCudaError(cudaFree(dev_numberToPassThroughDeviceMemory));
  checkCudaError(cudaFree(dev_productOfTheTwoNumbers));
}
