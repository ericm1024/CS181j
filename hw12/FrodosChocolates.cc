// -*- C++ -*-
// FrodosChocolates.cc
// cs181j hw12
// An example to illustrate how to use various functions in MPI for a white
//  elephant gift exchange in mordor

// magic header for all mpi stuff
#include <mpi.h>

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

// These utilities are used on many assignments
#include "../Utilities.h"

using std::string;
using std::vector;
using std::array;

int main(int argc, char* argv[]) {

  // ===============================================================
  // ********************** < Input> *******************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  const unsigned int maxExchanges  = 10000;
  const unsigned int rageQuitLimit = 100;

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </Input> *******************************
  // ===============================================================

  // make sure output directory exists
  Utilities::verifyThatDirectoryExists(std::string("data"));

  // Initialize MPI
  int mpiErrorCode = MPI_Init(&argc, &argv);
  if (mpiErrorCode != MPI_SUCCESS) {
    printf("error in MPI_Init; aborting...\n");
    exit(1);
  }

  // Figure out what rank I am
  int temp;
  MPI_Comm_rank(MPI_COMM_WORLD, &temp);
  const unsigned int rank = temp;
  MPI_Comm_size(MPI_COMM_WORLD, &temp);
  const unsigned int numberOfProcesses = temp;

  unsigned int exchangeIndex = 0;

  // TODO: run the simulation

  // TODO: obtain all chocolate-holding frequencies
  vector<unsigned int> globalChocolateFrequency(numberOfProcesses);

  // rank 0 outputs result
  if (rank == 0) {
    printf("we performed %6u exchanges\n", exchangeIndex);
    FILE* file = fopen("data/FrodosChocolates.csv", "w");
    printf("Chocolate Frequency:\n");
    unsigned int sumOfGlobalChocolateFrequency = 0;
    for (unsigned int processIndex = 0;
         processIndex < (unsigned)numberOfProcesses; ++processIndex) {
      if (processIndex > 0) {
        fprintf(file, ",");
      }
      fprintf(file, "%3u", globalChocolateFrequency[processIndex]);
      printf("%3u %3u : ", processIndex, globalChocolateFrequency[processIndex]);
      const unsigned int numberOfStars =
        globalChocolateFrequency[processIndex] /
        std::max(unsigned(1), rageQuitLimit / 100);
      for (unsigned int i = 0; i < numberOfStars; ++i) {
        printf("=");
      }
      printf("\n");
      sumOfGlobalChocolateFrequency += globalChocolateFrequency[processIndex];
    }
    fprintf(file, "\n");
    fclose(file);
    if (exchangeIndex != sumOfGlobalChocolateFrequency) {
      fprintf(stderr, "ERROR, exchangeIndex = %u != "
              "sumOfGlobalChocolateFrequency = %u\n",
              exchangeIndex, sumOfGlobalChocolateFrequency);
    }
  }
  MPI_Finalize();
  return 0;
}
