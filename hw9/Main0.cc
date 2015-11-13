// -*- C++ -*-
// Main0.cc
// cs181j hw9 primer
// cuda syntax!

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

// These utilities are used on many assignments
#include "../Utilities.h"

using std::string;
using std::vector;
using std::array;
using std::size_t;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

#include "Main0_cuda.cuh"

int main(int argc, char * argv[]) {

  // ===============================================================
  // ********************** < Input> *******************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  double numberToPassAsFunctionArgument = 9;
  if (argc > 1) {
    numberToPassAsFunctionArgument = atof(argv[1]);
  }

  double numberToPassThroughDeviceMemory = 17;
  if (argc > 1) {
    numberToPassThroughDeviceMemory = atof(argv[2]);
  }

  double productOfTheTwoNumbers = 0;
  sayHello_cuda(numberToPassAsFunctionArgument,
                &numberToPassThroughDeviceMemory,
                &productOfTheTwoNumbers);
  printf("the product of the two numbers %lf and %lf is %lf (cpu: %lf)\n",
         numberToPassAsFunctionArgument,
         numberToPassThroughDeviceMemory,
         productOfTheTwoNumbers,
         numberToPassAsFunctionArgument *
         numberToPassThroughDeviceMemory);

  return 0;
}
