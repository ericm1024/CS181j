// -*- C++ -*-
// MaximalIndependentSet.cc
// cs181j hw7
// An example of threading a graph algorithm

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

#include "Graphs.h"
#include "GraphUtilities.h"

#include "MaximalIndependentSet_functions.h"

using std::string;
using std::vector;
using std::array;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;


template <class Function>
void
runTest(const unsigned int numberOfTrials,
        const Function function,
        const std::string & testName,
        const unsigned int numberOfThreads,
        const Graphs::CompressedGraph & graph,
        double * const elapsedTime) {

  *elapsedTime = std::numeric_limits<double>::max();

  vector<unsigned int> independentSet;

  for (unsigned int trialNumber = 0;
       trialNumber < numberOfTrials; ++trialNumber) {

    independentSet.resize(0);

    // Start measuring
    const high_resolution_clock::time_point tic = high_resolution_clock::now();

    // Do the test
    function(numberOfThreads, graph, &independentSet);

    // Stop measuring
    const high_resolution_clock::time_point toc = high_resolution_clock::now();
    const double thisTrialsElapsedTime =
      duration_cast<duration<double> >(toc - tic).count();
    // Take the minimum values from all trials
    *elapsedTime = std::min(*elapsedTime, thisTrialsElapsedTime);

    try {
      // in case you didn't sort your set, I'll sort it.
      std::sort(independentSet.begin(), independentSet.end());
      GraphUtilities::checkIfSetIsAMaximalIndependentSetOfGraph(graph,
                                                                independentSet.begin(),
                                                                independentSet.end());
    } catch (const std::exception & e) {
      fprintf(stderr, "independent set was incorrect for version %s\n",
              testName.c_str());
      throw;
    }
  }

}

int main() {

  // ===========================================================================
  // *************************** < Inputs> *************************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  const array<double, 2> numberOfVerticesRange   = {{1e3, 1e5}};
  const unsigned int numberOfDataPoints          = 9;
  const vector<unsigned int> numbersOfThreads    =
    {{1, 2, 4, 6, 8, 10, 11, 12, 13, 14, 16, 20, 22, 24, 26, 28, 30, 36, 42, 48}};
  const unsigned int averageNumberOfNeighbors    = 10;
  const float averageNeighborCountVarianceFactor = 0.5;
  const unsigned int numberOfTrials              = 3;

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // *************************** </Inputs> *************************************
  // ===========================================================================

  const string prefix = "data/MaximalIndependentSet_";
  const string suffix = "_shuffler";

  // Make sure that the data directory exists.
  Utilities::verifyThatDirectoryExists("data");

  const unsigned randomSeed = 0;
  typedef std::default_random_engine RandomNumberEngine;
  RandomNumberEngine randomNumberEngine(randomSeed);

  vector<vector<double> >
    numberOfVerticesMatrixForPlotting(numberOfDataPoints,
                                      vector<double>(numbersOfThreads.size(), 0));
  vector<vector<double> >
    numberOfThreadsMatrixForPlotting(numberOfDataPoints,
                                     vector<double>(numbersOfThreads.size(), 0));
  vector<vector<double> >
    serialTimes(numberOfDataPoints,
                vector<double>(numbersOfThreads.size(), 0));
  vector<vector<double> >
    serialMaskTimes(numberOfDataPoints,
                    vector<double>(numbersOfThreads.size(), 0));
  vector<vector<double> >
    threadedMaskTimes(numberOfDataPoints,
                      vector<double>(numbersOfThreads.size(), 0));

  // for each numberOfVertices
  for (unsigned int dataPointIndex = 0;
       dataPointIndex < numberOfDataPoints;
       ++dataPointIndex) {

    const high_resolution_clock::time_point thisSizesTic =
      high_resolution_clock::now();

    const unsigned int numberOfVertices =
      Utilities::interpolateNumberLinearlyOnLogScale(numberOfVerticesRange[0],
                                                     numberOfVerticesRange[1],
                                                     numberOfDataPoints,
                                                     dataPointIndex);

    const unsigned int numberOfNeighborsVariance =
      std::max(unsigned(1),
               unsigned(averageNeighborCountVarianceFactor *
                        averageNumberOfNeighbors));
    // generate the graph
    const Graphs::VectorOfVectorsGraph vectorOfVectorsGraph =
      GraphUtilities::buildRandomGraph(numberOfVertices,
                                       averageNumberOfNeighbors,
                                       numberOfNeighborsVariance,
                                       &randomNumberEngine);

    // make a compressed version of it
    const Graphs::CompressedGraph graph =
      GraphUtilities::buildCompressedGraph(vectorOfVectorsGraph);

    double serialElapsedTime =
      std::numeric_limits<double>::quiet_NaN();
    const unsigned int numberOfThreadsForSerial = 1;
    runTest(numberOfTrials,
            findMaximalIndependentSet_serial,
            std::string("serial"),
            numberOfThreadsForSerial, // this is really ignored
            graph,
            &serialElapsedTime);

    double serialMaskElapsedTime =
      std::numeric_limits<double>::quiet_NaN();
    runTest(numberOfTrials,
            findMaximalIndependentSet_serialMask,
            std::string("serial mask"),
            numberOfThreadsForSerial, // this is really ignored
            graph,
            &serialMaskElapsedTime);

    // for each numberOfThreads
    for (unsigned int numberOfThreadsIndex = 0;
         numberOfThreadsIndex < numbersOfThreads.size();
         ++numberOfThreadsIndex) {
      // get the number of threads
      const unsigned int numberOfThreads =
        numbersOfThreads[numberOfThreadsIndex];

      try {

        // set the serial time
        serialTimes[dataPointIndex][numberOfThreadsIndex] =
          serialElapsedTime;
        serialMaskTimes[dataPointIndex][numberOfThreadsIndex] =
          serialMaskElapsedTime;

        // threaded version
        runTest(numberOfTrials,
                findMaximalIndependentSet_threadedMask,
                std::string("threadedMask"),
                numberOfThreads,
                graph,
                &threadedMaskTimes[dataPointIndex][numberOfThreadsIndex]);

      } catch (const std::exception & e) {
        fprintf(stderr, "error attempting %e vertices and %2u threads\n",
                float(numberOfVertices), numberOfThreads);
        throw;
      }


      numberOfVerticesMatrixForPlotting[dataPointIndex][numberOfThreadsIndex] =
        numberOfVertices;
      numberOfThreadsMatrixForPlotting[dataPointIndex][numberOfThreadsIndex] =
        numberOfThreads;
    }

    const high_resolution_clock::time_point thisSizesToc =
      high_resolution_clock::now();
    const double thisSizesElapsedTime =
      duration_cast<duration<double> >(thisSizesToc - thisSizesTic).count();
    printf("finished size %8.2e with %8.2e trials in %6.2f seconds\n",
           double(numberOfVertices), double(numberOfTrials),
           thisSizesElapsedTime);

  }

  Utilities::writeMatrixToFile(numberOfVerticesMatrixForPlotting,
                               prefix + string("numberOfVertices") + suffix);
  Utilities::writeMatrixToFile(numberOfThreadsMatrixForPlotting,
                               prefix + string("numberOfThreads") + suffix);
  Utilities::writeMatrixToFile(serialTimes, prefix + string("serial") + suffix);
  Utilities::writeMatrixToFile(serialMaskTimes, prefix + string("serialMask") + suffix);
  Utilities::writeMatrixToFile(threadedMaskTimes, prefix + string("threadedMask") + suffix);

  return 0;
}
