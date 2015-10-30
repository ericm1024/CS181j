// -*- C++ -*-
#ifndef GRAPH_UTILITIES_H
#define GRAPH_UTILITIES_H

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

#include "Graphs.h"

#include "../Utilities.h"

namespace GraphUtilities {

template <class Graph>
Graphs::CompressedGraph
buildCompressedGraph(const Graph & graph) {
  Graphs::CompressedGraph compressed_graph =
    Graphs::CompressedGraph(graph.getNumberOfVertices());

  unsigned int currDelimiter = 0;
  compressed_graph.delimiters_.push_back(currDelimiter);

  for (unsigned int vertex = 0; vertex < graph.getNumberOfVertices(); ++vertex) {
    currDelimiter += graph.getNumberOfNeighbors(vertex);
    compressed_graph.delimiters_.push_back(currDelimiter);

    std::vector<unsigned int>::const_iterator neighbor = graph.getNeighborhoodBegin(vertex);
    std::vector<unsigned int>::const_iterator end = graph.getNeighborhoodEnd(vertex);
    for ( ; neighbor != end; ++neighbor) {
      compressed_graph.neighborIndices_.push_back(*neighbor);
    }
  }
  return compressed_graph;
}

template <class RandomNumberEngine>
unsigned int
getUniformUnsignedInt(const unsigned int lowerBound,
                      const unsigned int upperBound,
                      RandomNumberEngine * randomNumberEngine) {
  std::uniform_int_distribution<unsigned int> nodeIndexGenerator(lowerBound,
                                                                 upperBound);
  return nodeIndexGenerator(*randomNumberEngine);
}

template <class RandomNumberEngine>
Graphs::VectorOfVectorsGraph
buildRandomGraph(const unsigned int numberOfVertices,
                 const unsigned int averageNumberOfNeighbors,
                 const unsigned int numberOfNeighborsVariance,
                 RandomNumberEngine * randomNumberEngine) {
  std::uniform_int_distribution<unsigned int> nodeIndexGenerator(0, numberOfVertices-1);
  std::normal_distribution<double> numberOfNeighborsGenerator(double(averageNumberOfNeighbors),
                                                              std::sqrt(double(numberOfNeighborsVariance)));
  // initialize graph with no edges
  Graphs::VectorOfVectorsGraph graph;
  graph.neighborLists_.resize(numberOfVertices);

  // make the target degrees with the average and variance
  std::vector<unsigned int> targetDegrees(numberOfVertices);
  for (unsigned int vertex = 0; vertex < numberOfVertices; ++vertex) {
    // have to max with 0 or else we wrap around with unsigned ints
    const unsigned int desiredNumberOfNeighbors =
      std::round(std::max(0., numberOfNeighborsGenerator(*randomNumberEngine)));
    // make sure it's not more than the number of vertices - it's already no
    //  fewer than zero because unsigned
    targetDegrees[vertex] =
      std::min(desiredNumberOfNeighbors, numberOfVertices - 1);
  }

  // start all degrees at zero
  std::vector<unsigned int> degrees(numberOfVertices, 0);
  // start all vertices as needing neighbors
  std::vector<unsigned int> activeVertices;
  activeVertices.reserve(numberOfVertices);
  for (unsigned int vertex = 0; vertex < numberOfVertices; ++vertex) {
    if (targetDegrees[vertex] > 0) {
      activeVertices.push_back(vertex);
    }
  }

  std::vector<unsigned int> copyOfActiveVertices;
  unsigned int iteration = 0;
  while (activeVertices.size() > 1) {
    ++iteration;
    const unsigned int indexOfStartVertex =
      getUniformUnsignedInt(0, activeVertices.size() - 1, randomNumberEngine);
    const unsigned int startVertex = activeVertices[indexOfStartVertex];


    unsigned int neighborVertex = std::numeric_limits<unsigned int>::max();

    // take a stab in the dark with the other neighbor vertex.  if it works, we
    //  don't worry about doing the expensive thing below.  if it doesn't, we'll
    //  do the expensive thing.
    {
      const unsigned int randomNeighborNumber =
        getUniformUnsignedInt(0, activeVertices.size() - 1,
                              randomNumberEngine);
      const unsigned int possibleEndVertex =
        activeVertices[randomNeighborNumber];
      // check if this edge already exists
      const bool thisEdgeAlreadyExists =
        std::binary_search(graph.neighborLists_[startVertex].begin(),
                           graph.neighborLists_[startVertex].end(),
                           possibleEndVertex);
      // if it's not a loop and not a repeated edge, we're ready to use it
      if ((possibleEndVertex == startVertex ||
           thisEdgeAlreadyExists == true) == false) {
        neighborVertex = possibleEndVertex;
      }
    }

    if (neighborVertex == std::numeric_limits<unsigned int>::max()) {
      // make a copy of the indices, ouch
      copyOfActiveVertices = activeVertices;
      // while we haven't found a neighbor
      while (neighborVertex == std::numeric_limits<unsigned int>::max() &&
             copyOfActiveVertices.size() > 0) {
        const unsigned int randomNeighborNumber =
          getUniformUnsignedInt(0, copyOfActiveVertices.size() - 1,
                                randomNumberEngine);
        const unsigned int possibleEndVertex =
          activeVertices[randomNeighborNumber];
        if (possibleEndVertex == startVertex) {
          // it's a loop, get rid of this number
          copyOfActiveVertices.erase(copyOfActiveVertices.begin() +
                                     randomNeighborNumber);
          // keep going in the loop
          continue;
        }
        // check if this edge already exists
        const bool thisEdgeAlreadyExists =
          std::binary_search(graph.neighborLists_[startVertex].begin(),
                             graph.neighborLists_[startVertex].end(),
                             possibleEndVertex);
        if (thisEdgeAlreadyExists == true) {
          // if the edge already exists, eliminate this from the candidates
          copyOfActiveVertices.erase(copyOfActiveVertices.begin() +
                                     randomNeighborNumber);
          // keep going in the loop
          continue;
        }
        // the edge doesn't exist and it's not a loop, so we accept the possible
        //  end vertex
        neighborVertex = possibleEndVertex;
      }
    }

    // if we didn't find anything, remove the startVertex
    if (neighborVertex == std::numeric_limits<unsigned int>::max()) {
      activeVertices.erase(activeVertices.begin() +
                           indexOfStartVertex);
    } else {
      // we did find something, so add the edge.
      std::vector<unsigned int>::iterator insertLocation =
        graph.neighborLists_[startVertex].begin();
      std::vector<unsigned int>::iterator end =
        graph.neighborLists_[startVertex].end();
      insertLocation = std::lower_bound(insertLocation, end, neighborVertex);
      // add neighbor, maintaining sorted order for binary search
      graph.neighborLists_[startVertex].insert(insertLocation, neighborVertex);
      ++degrees[startVertex];
      if (degrees[startVertex] == targetDegrees[startVertex]) {
        // remove startVertex from active vertices
        activeVertices.erase(activeVertices.begin() +
                             indexOfStartVertex);
      }
      // also add reverse edge
      insertLocation = graph.neighborLists_[neighborVertex].begin();
      end = graph.neighborLists_[neighborVertex].end();
      insertLocation = std::lower_bound(insertLocation, end, startVertex);
      // add neighbor, maintaining sorted order for binary search
      graph.neighborLists_[neighborVertex].insert(insertLocation, startVertex);
      ++degrees[neighborVertex];
      if (degrees[neighborVertex] == targetDegrees[neighborVertex]) {
        // to remove the neighborVertex from the active vertices, it may
        //  have changed position so we have to find it again.
        const auto equal_range_results =
          std::equal_range(activeVertices.begin(),
                           activeVertices.end(),
                           neighborVertex);
        if (equal_range_results.first == activeVertices.end()) {
          throwException("could not find iterator to remove %u from the "
                         "activeVertices", neighborVertex);
        }
        // remove startVertex from active vertices
        activeVertices.erase(equal_range_results.first);
      }
#if 0
      const bool startIsSorted =
        std::is_sorted(graph.neighborLists_[startVertex].begin(),
                       graph.neighborLists_[startVertex].end());
      if (startIsSorted == false) {
        throwException("start of %u is not sorted after insertion of %u on "
                       "iteration %u\n",
                       startVertex, neighborVertex, iteration);
      }
      const bool neighborIsSorted =
        std::is_sorted(graph.neighborLists_[neighborVertex].begin(),
                       graph.neighborLists_[neighborVertex].end());
      if (startIsSorted == false) {
        throwException("start of %u is not sorted after insertion of %u on "
                       "iteration %u\n",
                       startVertex, neighborVertex, iteration);
      }
#endif
    }
  }

  unsigned int numberOfInconsistent = 0;
  for (unsigned int vertex = 0; vertex < numberOfVertices; ++vertex) {
    if (std::abs(int(degrees[vertex]) - int(targetDegrees[vertex])) > 1) {
      ++numberOfInconsistent;
    }
    const unsigned int numberOfNeighbors =
      graph.neighborLists_[vertex].size();
    if (degrees[vertex] != numberOfNeighbors) {
      throwException("bad degrees[%u] = %u when numberOfNeighbors = %u",
                     vertex, degrees[vertex], numberOfNeighbors);
    }
  }
  const double fractionOfInconsistent =
    numberOfInconsistent / double(numberOfVertices);
  if (numberOfInconsistent > 5 && fractionOfInconsistent > 0.01) {
    warning("found a high number of vertices which didn't get their "
            "target degrees: %u / %u (%%%5.2lf)\n",
            numberOfInconsistent, numberOfVertices,
            100 * fractionOfInconsistent);
  }
  if (numberOfVertices != graph.getNumberOfVertices()) {
    throwException("buildRandomGraph did not make as many vertices (%u) "
                   "as were requested (%u)",
                   graph.getNumberOfVertices(),
                   numberOfVertices);
  }

  return graph;
}

// assumes independentSet is sorted
template <class Graph, class IndependentSetIterator>
void
checkIfSetIsAMaximalIndependentSetOfGraph(const Graph & graph,
                                          const IndependentSetIterator independentSetBegin,
                                          const IndependentSetIterator independentSetEnd) {

  unsigned int numberOfVertices = graph.getNumberOfVertices();
  for (unsigned int vertex = 0; vertex < numberOfVertices; ++vertex) {
    std::vector<unsigned int>::const_iterator neighbor =
      graph.getNeighborhoodBegin(vertex);
    const std::vector<unsigned int>::const_iterator end =
      graph.getNeighborhoodEnd(vertex);

    if (std::binary_search(independentSetBegin, independentSetEnd, vertex)) {
      // no element of the set can have neighbors in the set
      for ( ; neighbor != end; ++neighbor) {
        if (std::binary_search(independentSetBegin, independentSetEnd, *neighbor)) {
          throwException("set is not independent, vertex %u and neighbor %u "
                         "are both in set", vertex, *neighbor);
        }
      }
    } else {
      // each element not in the set must have a neighbor in the set
      bool couldBeAdded = true;
      for ( ; neighbor != end; ++neighbor) {
        if (std::binary_search(independentSetBegin, independentSetEnd, *neighbor)) {
          couldBeAdded = false;
          break;
        }
      }

      if (couldBeAdded) {
        throwException("set is not maximal, vertex %u could still be added", vertex);
      }
    }
  }
}

}

#endif // GRAPH_UTILITIES_H
