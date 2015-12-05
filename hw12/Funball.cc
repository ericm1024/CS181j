// -*- C++ -*-
// Funball.cc
// cs181j hw12
// An example to illustrate communication patterns in the aftermath of funball

// magic header for all mpi stuff
#include <mpi.h>

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

// header for the magic black box functions that we're using on this assignment
#include "Funball.h"

#include <map>
#include <cstddef>

using std::string;
using std::vector;
using std::array;
using std::map;

int main(int argc, char* argv[]) {

  // ===============================================================
  // ********************** < Input> *******************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  const unsigned int numberOfStudentsPerDorm  = 1000;

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </Input> *******************************
  // ===============================================================

  // Initialize MPI
  int mpiErrorCode = MPI_Init(&argc, &argv);
  if (mpiErrorCode != MPI_SUCCESS) {
    printf("error in MPI_Init; aborting...\n");
    exit(1);
  }

  // At some point in time, you'll need to send MuddMoneys to other ranks.
  // You can either stuff the mudd moneys into an array of doubles and send it,
  //  or you can send an array of MuddMoneys by using this MPI_MUDDMONEY
  //  data type defined here.
  const unsigned int numberOfItemsInAMuddMoney = 3;
  int itemLengths[numberOfItemsInAMuddMoney] = {1,1,1};
  MPI_Datatype types[3] = {MPI_UNSIGNED, MPI_UNSIGNED, MPI_DOUBLE};
  MPI_Aint     offsets[3];
  offsets[0] = offsetof(Funball::MuddMoney, _idWithinTheDorm);
  offsets[1] = offsetof(Funball::MuddMoney, _dormNumber);
  offsets[2] = offsetof(Funball::MuddMoney, _amountOfMoney);
  MPI_Datatype MPI_MUDDMONEY;
  MPI_Type_create_struct(numberOfItemsInAMuddMoney, itemLengths, offsets, types,
                         &MPI_MUDDMONEY);

  MPI_Type_commit(&MPI_MUDDMONEY);

  // Figure out what rank I am
  int temp;
  MPI_Comm_rank(MPI_COMM_WORLD, &temp);
  const unsigned int rank = temp;
  MPI_Comm_size(MPI_COMM_WORLD, &temp);
  const unsigned int numberOfProcesses = temp;

  const unsigned long long randomNumberSeed = time(0);

  const vector<Funball::MuddMoney> thisRanksStartingMuddMoneys =
    Funball::initializeFun(numberOfStudentsPerDorm,
                           randomNumberSeed);

  // See the definition of MuddMoney in Funball.h
  vector<Funball::MuddMoney> thisRanksFinalMuddMoneys;

  // TODO: populate thisRanksFinalMuddMoneys

  // check the answer
  Funball::checkFunballAnswer(numberOfStudentsPerDorm,
                              thisRanksStartingMuddMoneys,
                              thisRanksFinalMuddMoneys);

  MPI_Type_free(&MPI_MUDDMONEY);

  MPI_Finalize();
  return 0;
}
