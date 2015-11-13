// -*- C++ -*-
#ifndef MAIN0_CUDA_CUH
#define MAIN0_CUDA_CUH


void
sayHello_cuda(const double numberToPassAsFunctionArgument,
              const double * const numberToPassThroughDeviceMemory,
              double * productOfTheTwoNumbers);

#endif // MAIN0_CUDA_CUH
