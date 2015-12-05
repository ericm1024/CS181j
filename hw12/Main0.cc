// -*- C++ -*-
// Main0.cc
// cs181j hw12 sample syntax

// c junk
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <unistd.h>

// c++ junk
#include <vector>
#include <algorithm>
#include <random>
#include <string>
#include <array>
#include <fstream>

using std::string;
using std::vector;
using std::array;

// magic header for all mpi stuff
#include <mpi.h>


int main(int argc, char* argv[]) {

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
  if (numberOfProcesses == 1) {
    printf("Remember, to run this with mpi, you need to do "
           "\"mpirun -np X ./Main0\"\n");
  }


  const double baseTime = MPI_Wtime();

  // sum the ranks, have rank 0 output it
  unsigned int totalRanks;
  MPI_Reduce(&rank, &totalRanks, 1, MPI_UNSIGNED, MPI_SUM, 0,
             MPI_COMM_WORLD);
  if (rank == 0) {
    printf("Rank %2u, reports totalRanks of %d from MPI_Reduce\n",
           rank, totalRanks);
  }

  // everyone stops here
  MPI_Barrier(MPI_COMM_WORLD);



  // sum the ranks, have all ranks output it
  MPI_Allreduce(&rank, &totalRanks, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  printf("Rank %2u, reports totalRanks of %d from MPI_Allreduce\n",
         rank, totalRanks);

  // everyone stops here
  MPI_Barrier(MPI_COMM_WORLD);




  // gather random numbers
  srand(time(0) + rank);
  unsigned int randomNumber = rand();
  vector<unsigned int> randomNumbers(numberOfProcesses);
  MPI_Gather(&randomNumber, 1, MPI_UNSIGNED, &randomNumbers[0], 1, MPI_UNSIGNED,
             0, MPI_COMM_WORLD);
  if (rank == 0) {
    printf("Gathered randomNumbers:");
    for (unsigned int processIndex = 0;
         processIndex < unsigned(numberOfProcesses); ++processIndex) {
      printf(" %u", randomNumbers[processIndex]);
    }
    printf("\n");
  }

  // everyone stops here
  MPI_Barrier(MPI_COMM_WORLD);




  float thisRanksFloat = rand() / float(RAND_MAX);
  float otherRanksFloat;

  float rank0sFloat = thisRanksFloat;
  MPI_Bcast(&rank0sFloat, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  printf("Rank %2u, from Bcast rank 0's float is %f\n", rank, rank0sFloat);

  // everyone stops here
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    printf("\nDoing synchronous send, synchronous receive\n");
  }

  int tag = 29;   // i just turned it!!1!

  if (numberOfProcesses == 1) {
    printf("Only 1 process used, so this blocking communication will die...\n");
  }
  if (rank == 0) {
    printf("Rank %2u, trying to send 1 float to rank %d : %f\n",
           rank, numberOfProcesses - 1, thisRanksFloat);
    printf("Rank %2u, MPI_Ssend started at   %lf\n", rank, MPI_Wtime() - baseTime);
    MPI_Ssend(&thisRanksFloat, 1, MPI_FLOAT, numberOfProcesses - 1, tag,
              MPI_COMM_WORLD);
    printf("Rank %2u, MPI_Ssend completed at %lf\n", rank, MPI_Wtime() - baseTime);
  }
  if (rank == numberOfProcesses - 1) {
    printf("Rank %2u, going to sleep for 5 seconds\n", rank);
    sleep(5);
    MPI_Status status;
    printf("Rank %2u, MPI_Recv started at   %lf\n", rank, MPI_Wtime() - baseTime);
    MPI_Recv(&otherRanksFloat, 1, MPI_FLOAT, 0, tag, MPI_COMM_WORLD,
             &status);
    printf("Rank %2u, MPI_Recv completed at %lf\n", rank, MPI_Wtime() - baseTime);
    int floatsReceived;
    MPI_Get_count(&status, MPI_FLOAT, &floatsReceived);
    printf("Rank %2u, received %u float from rank 0: %f\n",
           numberOfProcesses - 1, floatsReceived, otherRanksFloat);
  }

  // everyone stops here
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    printf("\nDoing asynchronous send, synchronous receive\n");
  }
  thisRanksFloat = rand() / float(RAND_MAX);

  MPI_Request sendRequest;
  if (rank == 0) {
    printf("Rank %2u, trying to send 1 float to rank %d : %f\n",
           rank, numberOfProcesses - 1, thisRanksFloat);
    printf("Rank %2u, MPI_Issend started at   %lf\n", rank, MPI_Wtime() - baseTime);
    MPI_Issend(&thisRanksFloat, 1, MPI_FLOAT, numberOfProcesses - 1, tag,
               MPI_COMM_WORLD, &sendRequest);
    printf("Rank %2u, MPI_Issend completed at %lf\n", rank, MPI_Wtime() - baseTime);
  }
  if (rank == numberOfProcesses - 1) {
    printf("Rank %2u, going to sleep for 5 seconds\n", rank);
    sleep(5);
    MPI_Status status;
    printf("Rank %2u, MPI_Recv started at   %lf\n", rank, MPI_Wtime() - baseTime);
    MPI_Recv(&otherRanksFloat, 1, MPI_FLOAT, 0, tag, MPI_COMM_WORLD,
             &status);
    printf("Rank %2u, MPI_Recv completed at %lf\n", rank, MPI_Wtime() - baseTime);
    int floatsReceived;
    MPI_Get_count(&status, MPI_FLOAT, &floatsReceived);
    printf("Rank %2u, received %u float from rank 0: %f\n",
           rank, floatsReceived, otherRanksFloat);
  }
  if (rank == 0) {
    MPI_Status sendStatus;
    MPI_Wait(&sendRequest, &sendStatus);
  }

  // everyone stops here
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    printf("\nDoing asynchronous send, asynchronous receive\n");
  }
  thisRanksFloat = rand() / float(RAND_MAX);

  if (rank == 0) {
    printf("Rank %2u, trying to send 1 float to rank %d : %f\n",
           rank, numberOfProcesses - 1, thisRanksFloat);
    printf("Rank %2u, MPI_Issend started at   %lf\n", rank, MPI_Wtime() - baseTime);
    MPI_Issend(&thisRanksFloat, 1, MPI_FLOAT, numberOfProcesses - 1, tag,
               MPI_COMM_WORLD, &sendRequest);
    printf("Rank %2u, MPI_Issend completed at %lf\n", rank, MPI_Wtime() - baseTime);
  }
  MPI_Request recvRequest;
  if (rank == numberOfProcesses - 1) {
    // NOTE the use of MPI_ANY_SOURCE
    printf("Rank %2u, MPI_Irecv started at   %lf\n", rank, MPI_Wtime() - baseTime);
    MPI_Irecv(&otherRanksFloat, 1, MPI_FLOAT, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD,
              &recvRequest);
    printf("Rank %2u, MPI_Irecv completed at %lf\n", rank, MPI_Wtime() - baseTime);
    printf("Rank %2u, going to sleep for 5 seconds\n", rank);
    sleep(5);
  }
  if (rank == 0) {
    MPI_Status sendStatus;
    MPI_Wait(&sendRequest, &sendStatus);
  }
  if (rank == numberOfProcesses - 1) {
    MPI_Status recvStatus;
    MPI_Wait(&recvRequest, &recvStatus);
    int floatsReceived;
    MPI_Get_count(&recvStatus, MPI_FLOAT, &floatsReceived);
    // NOTE the use of MPI_SOURCE to find out who sent us the thing
    printf("Rank %2u, received %u float from rank %u: %lf\n",
           rank, floatsReceived, recvStatus.MPI_SOURCE, otherRanksFloat);
  }

  // say goodbye
  MPI_Finalize();
  return 0;
}
