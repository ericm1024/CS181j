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

        for (unsigned int row = 0; row < matrixSize; ++row) {
                for (unsigned int col = 0; col < matrixSize; ++col) {
                        rowMajorResultMatrix[row * matrixSize + col] = 0;
                        for (unsigned int k = 0; k < matrixSize; ++k) {
                                rowMajorResultMatrix[row * matrixSize + col]
                                        += colMajorLeftMatrix[row + k * matrixSize]
                                        * rowMajorRightMatrix[k * matrixSize + col];
                        }
                }
        }
}

void
multiplyRowMajorByColMajorMatrices(const unsigned int matrixSize,
                                   const vector<double> & rowMajorLeftMatrix,
                                   const vector<double> & colMajorRightMatrix,
                                   vector<double> * rowMajorResultMatrix_pointer) {

        vector<double> & rowMajorResultMatrix = *rowMajorResultMatrix_pointer;

        for (unsigned int row = 0; row < matrixSize; ++row) {
                for (unsigned int col = 0; col < matrixSize; ++col) {
                        rowMajorResultMatrix[row * matrixSize + col] = 0;
                        for (unsigned int k = 0; k < matrixSize; ++k) {
                                rowMajorResultMatrix[row * matrixSize + col]
                                        += rowMajorLeftMatrix[row * matrixSize + k]
                                        * colMajorRightMatrix[k + col * matrixSize];
                        }
                }
        }

}

void
multiplyRowMajorByColMajorMatrices_improved(const unsigned int matrixSize,
                                            const vector<double> & rowMajorLeftMatrix,
                                            const vector<double> & colMajorRightMatrix,
                                            vector<double> * rowMajorResultMatrix_pointer) {

        vector<double> & rowMajorResultMatrix = *rowMajorResultMatrix_pointer;

        for (unsigned int row = 0; row < matrixSize; ++row) {
                for (unsigned int col = 0; col < matrixSize; ++col) {
                        double tmp = 0;
                        for (unsigned int k = 0; k < matrixSize; ++k) {
                                tmp += rowMajorLeftMatrix[row * matrixSize + k]
                                        * colMajorRightMatrix[k + col * matrixSize];
                        }
                        rowMajorResultMatrix[row * matrixSize + col] = tmp;
                }
        }
}

}

#endif // MAIN2_FUNCTIONS_H
