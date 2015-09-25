// -*- C++ -*-
#ifndef MAIN1_FUNCTORS_H
#define MAIN1_FUNCTORS_H

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

struct MatrixMultiplier_ColMajorByRowMajor {
  const unsigned int _matrixSize;

  MatrixMultiplier_ColMajorByRowMajor(const unsigned int matrixSize) :
    _matrixSize(matrixSize) {
  }

  void multiplyMatrices(const std::vector<double> & colMajorLeftMatrix,
                        const std::vector<double> & rowMajorRightMatrix,
                        std::vector<double> * rowMajorResultMatrix_pointer) const {

    std::vector<double> & rowMajorResultMatrix = *rowMajorResultMatrix_pointer;

    for (unsigned int row = 0; row < _matrixSize; ++row) {
      for (unsigned int col = 0; col < _matrixSize; ++col) {
        const unsigned int resultIndex = row * _matrixSize + col;
        rowMajorResultMatrix[resultIndex] = 0;
        for (unsigned int k = 0; k < _matrixSize; ++k) {
          rowMajorResultMatrix[resultIndex] +=
            colMajorLeftMatrix[row + k * _matrixSize] *
            rowMajorRightMatrix[k * _matrixSize + col];
        }
      }
    }
  }

};

struct MatrixMultiplier_RowMajorByColMajor {
  const unsigned int _matrixSize;

  MatrixMultiplier_RowMajorByColMajor(const unsigned int matrixSize) :
    _matrixSize(matrixSize) {
  }

  void multiplyMatrices(const std::vector<double> & rowMajorLeftMatrix,
                        const std::vector<double> & colMajorRightMatrix,
                        std::vector<double> * rowMajorResultMatrix_pointer) const {

    std::vector<double> & rowMajorResultMatrix = *rowMajorResultMatrix_pointer;

    for (unsigned int row = 0; row < _matrixSize; ++row) {
      for (unsigned int col = 0; col < _matrixSize; ++col) {
        const unsigned int resultIndex = row * _matrixSize + col;
        rowMajorResultMatrix[resultIndex] = 0;
        for (unsigned int k = 0; k < _matrixSize; ++k) {
          rowMajorResultMatrix[resultIndex] +=
            rowMajorLeftMatrix[row * _matrixSize + k] *
            colMajorRightMatrix[k + col * _matrixSize];
        }
      }
    }
  }

};

struct MatrixMultiplier_RowMajorByColMajor_Improved {
  const unsigned int _matrixSize;

  MatrixMultiplier_RowMajorByColMajor_Improved(const unsigned int matrixSize) :
    _matrixSize(matrixSize) {
  }

  void multiplyMatrices(const std::vector<double> & rowMajorLeftMatrix,
                        const std::vector<double> & colMajorRightMatrix,
                        std::vector<double> * rowMajorResultMatrix_pointer) const {

    std::vector<double> & rowMajorResultMatrix = *rowMajorResultMatrix_pointer;

    static const unsigned int NUMBER_OF_BUCKETS = 8;
    std::array<double, NUMBER_OF_BUCKETS> sums;
    for (unsigned int row = 0; row < _matrixSize; ++row) {
      for (unsigned int col = 0; col < _matrixSize; ++col) {
        sums.fill(0);
        unsigned int k = 0;
        for (; k < _matrixSize - NUMBER_OF_BUCKETS; k += NUMBER_OF_BUCKETS) {
          const unsigned int rowIndex = row * _matrixSize + k;
          const unsigned int colIndex = col * _matrixSize + k;
          for (unsigned int bucketIndex = 0;
               bucketIndex < NUMBER_OF_BUCKETS; ++bucketIndex) {
            sums[bucketIndex] += rowMajorLeftMatrix[rowIndex + bucketIndex] *
              colMajorRightMatrix[colIndex + bucketIndex];
          }
        }
        for (; k < _matrixSize; ++k) {
          sums[0] += rowMajorLeftMatrix[row * _matrixSize + k] *
            colMajorRightMatrix[k + col * _matrixSize];
        }
        for (unsigned int bucketIndex = 1;
             bucketIndex < NUMBER_OF_BUCKETS; ++bucketIndex) {
          sums[0] += sums[bucketIndex];
        }
        rowMajorResultMatrix[row * _matrixSize + col] = sums[0];
      }
    }
  }

};

// STUDENTS: THIS IS THE STRUCT YOU HAVE TO MODIFY.
struct MatrixMultiplier_RowMajorByRowMajor_Tiled {
  const unsigned int _matrixSize;
  const unsigned int _tileSize;

  MatrixMultiplier_RowMajorByRowMajor_Tiled(const unsigned int matrixSize,
                                            const unsigned int tileSize) :
    _matrixSize(matrixSize), _tileSize(tileSize) {
    // TODO: do you need to store or calculate more stuff?
  }

  void multiplyMatrices(const std::vector<double> & rowMajorLeftMatrix,
                        const std::vector<double> & rowMajorRightMatrix,
                        std::vector<double> * rowMajorResultMatrix_pointer) const {

    std::vector<double> & rowMajorResultMatrix = *rowMajorResultMatrix_pointer;

    // TODO: write the tiled matrix multiplication algorithm
    throw std::runtime_error("You haven't yet implemented the tiled "
                             "multiplication version!");

  }
};

#endif // MAIN1_FUNCTORS_H
