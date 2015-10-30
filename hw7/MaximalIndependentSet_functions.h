// -*- C++ -*-
#ifndef MAXIMAL_INDEPENDENT_SET_FUNCTIONS_H
#define MAXIMAL_INDEPENDENT_SET_FUNCTIONS_H

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

// These utilities are used on many assignments
#include "../Utilities.h"

#include "Graphs.h"
#include "GraphUtilities.h"

#include <set>

#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

// header file for openmp
#include <omp.h>

void
findMaximalIndependentSet_serial(const unsigned int ignoredNumberOfThreads,
                                 const Graphs::CompressedGraph & graph,
                                 std::vector<unsigned int> * independentSet) {
  ignoreUnusedVariable(ignoredNumberOfThreads);

  const unsigned int numberOfVertices = graph.getNumberOfVertices();

  // All vertices are candidates at first.
  std::set<unsigned int> candidates;
  for (unsigned int vertexIndex = 0;
       vertexIndex < numberOfVertices; ++vertexIndex) {
    candidates.insert(vertexIndex);
  }

  // While there are still candidate vertices to either add to the
  //  independent set or remove from consideration.
  while (candidates.size() > 0) {

    // This loop is weird because we are iterating over a set, and on some
    //  iterations we remove things from the set.  So, it's complicated.
    // When you convert this to use a mask, you don't need this type of
    //  complicated loop.
    std::set<unsigned int>::iterator candidateIter = candidates.begin();
    while (candidateIter != candidates.end()) {
      const unsigned int candidateIndex = *candidateIter;
      bool thisVertexShouldBeAddedToIndependentSet = true;
      // If we have the largest vertex index out of our candidate neighbors,
      //  we add ourselves.
      const std::vector<unsigned int>::const_iterator neighborhoodEnd =
        graph.getNeighborhoodEnd(candidateIndex);
      for (std::vector<unsigned int>::const_iterator neighborIter =
             graph.getNeighborhoodBegin(candidateIndex);
           neighborIter != neighborhoodEnd; ++neighborIter) {
        const unsigned int neighborIndex = *neighborIter;
        // If the neighbor has a higher vertex index than we do
        if (neighborIndex > candidateIndex &&
            //  and the neighbor is a candidate
            candidates.find(neighborIndex) != candidates.end()) {
          // then they win and we don't add ourselves.
          thisVertexShouldBeAddedToIndependentSet = false;
          break;
        }
      }

      if (thisVertexShouldBeAddedToIndependentSet == true) {
        // We add ourselves to the independent set
        independentSet->push_back(candidateIndex);
        // We remove all of our neighbors from the candidate vertices,
        //  if they are in it.
        for (std::vector<unsigned int>::const_iterator neighborIter =
               graph.getNeighborhoodBegin(candidateIndex);
             neighborIter != neighborhoodEnd; ++neighborIter) {
          candidates.erase(*neighborIter);
        }
        // Now, we remove ourselves from the candidate vertices.
        // This is super weird, and you won't need to do it for the mask
        //  version.
        // We save an iterator for what to erase (our current iterator),
        //  we increment the iterator that we're keeping, and then we erase
        //  the iterator we saved.
        const auto temp = candidateIter;
        ++candidateIter;
        candidates.erase(temp);
      } else {
        // If we don't add ourselves, we just keep going on in the loop
        ++candidateIter;
      }
    }
  }
}

void
findMaximalIndependentSet_serialMask(const unsigned int ignoredNumberOfThreads,
                                     const Graphs::CompressedGraph & graph,
                                     std::vector<unsigned int> * independentSet) {
  ignoreUnusedVariable(ignoredNumberOfThreads);

  // Reminder: don't use a vector<bool>, use a "bool *"

  // TODO: delete this call and replace with a serial mask version
  findMaximalIndependentSet_serial(ignoredNumberOfThreads,
                                   graph,
                                   independentSet);
}

void
findMaximalIndependentSet_threadedMask(const unsigned int numberOfThreads,
                                       const Graphs::CompressedGraph & graph,
                                       std::vector<unsigned int> * independentSet) {
  omp_set_num_threads(numberOfThreads);

  // Reminder: don't use a vector<bool>, use a "bool *"

  // TODO: delete this call and replace with a threaded mask version
  findMaximalIndependentSet_serialMask(numberOfThreads,
                                       graph,
                                       independentSet);

}

#endif // MAXIMAL_INDEPENDENT_SET_FUNCTIONS_H
