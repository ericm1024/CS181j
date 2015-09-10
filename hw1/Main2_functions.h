// -*- C++ -*-
#ifndef MAIN2_FUNCTIONS_H
#define MAIN2_FUNCTIONS_H

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

// Only bring in from the standard namespace things that we care about.
// Remember, it's a naughty thing to just use the whole namespace.
using std::vector;
using std::array;

namespace Main2 {

void
multiplyColMajorByRowMajorMatrices(const unsigned int matrixSize,
                                   const vector<double> & colMajorLeftMatrix,
                                   const vector<double> & rowMajorRightMatrix,
                                   vector<double> * rowMajorResultMatrix_pointer) {

  vector<double> & rowMajorResultMatrix = *rowMajorResultMatrix_pointer;

  // TODO

}

void
multiplyRowMajorByColMajorMatrices(const unsigned int matrixSize,
                                   const vector<double> & rowMajorLeftMatrix,
                                   const vector<double> & colMajorRightMatrix,
                                   vector<double> * rowMajorResultMatrix_pointer) {

  vector<double> & rowMajorResultMatrix = *rowMajorResultMatrix_pointer;

  // TODO

}

void
multiplyRowMajorByColMajorMatrices_improved(const unsigned int matrixSize,
                                            const vector<double> & rowMajorLeftMatrix,
                                            const vector<double> & colMajorRightMatrix,
                                            vector<double> * rowMajorResultMatrix_pointer) {

  vector<double> & rowMajorResultMatrix = *rowMajorResultMatrix_pointer;

  // TODO

}

}

#endif // MAIN2_FUNCTIONS_H
