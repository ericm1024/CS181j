// -*- C++ -*-
#ifndef GRAPHS_H
#define GRAPHS_H

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

// forward declare these because they are needed for the following function
//  declarations which are needed for the class definitions.  yikes.
namespace Graphs {

class CompressedGraph;
class VectorOfVectorsGraph;

}

// declare these functions because they need to be friends
namespace GraphUtilities {

template <class Graph>
Graphs::CompressedGraph
buildCompressedGraph(const Graph & graph);

template <class RandomNumberEngine>
Graphs::VectorOfVectorsGraph
buildRandomGraph(const unsigned int numberOfVertices,
                 const unsigned int averageNumberOfNeighbors,
                 const unsigned int numberOfNeighborsVariance,
                 RandomNumberEngine * randomNumberEngine);

}

namespace Graphs {

class CompressedGraph {

  CompressedGraph(const unsigned int numberOfVertices) :
    numberOfVertices_(numberOfVertices) {
  }

  const unsigned int numberOfVertices_;
  std::vector<unsigned int> neighborIndices_;
  std::vector<unsigned int> delimiters_;

  void
  checkConsistency() const {
    if (numberOfVertices_ + 1 != delimiters_.size()) {
      throwException("for a graph with %u vertices, there should be %u "
                     "delimiters, but there are %zu",
                     numberOfVertices_, numberOfVertices_ + 1,
                     delimiters_.size());
    }
    if (delimiters_.size() < 1) {
      throwException("invalid number of delimiters = %zu, numberOfVertices_ "
                     "= %u", delimiters_.size(), numberOfVertices_);
    }
    if (delimiters_[0] != 0) {
      throwException("invalid starting delimiter = %u", delimiters_[0]);
    }
    if (delimiters_.back() != neighborIndices_.size()) {
      throwException("invalid last delimiter = %u, should be the size of the "
                     "neighborIndices_, which is %zu", delimiters_.back(),
                     neighborIndices_.size());
    }
  }

  void
  checkVertexIndex(const unsigned int vertexIndex) const {
    checkConsistency();
    if (vertexIndex >= numberOfVertices_) {
      throwException("invalid vertexIndex = %u for graph with only "
                     "%u vertices", vertexIndex, numberOfVertices_);
    }
  }

  void
  checkNeighborNumber(const unsigned int vertexIndex,
                      const unsigned int neighborNumber) const {
    checkConsistency();
    checkVertexIndex(vertexIndex);
    const unsigned int numberOfNeighbors =
      delimiters_[vertexIndex + 1] - delimiters_[vertexIndex];
    if (neighborNumber >= numberOfNeighbors) {
      throwException("invalid neighborNumber = %u for vertex %u which has %u "
                     "neighbors.", neighborNumber, vertexIndex,
                     numberOfNeighbors);
    }
  }

public:

  typedef std::vector<unsigned int>::const_iterator    const_iterator;

  unsigned int
  getNumberOfNeighbors(const unsigned int vertexIndex) const {
#ifdef DEBUG_MODE
    checkVertexIndex(vertexIndex);
#endif
    return delimiters_[vertexIndex + 1] - delimiters_[vertexIndex];
  }

  unsigned int
  getNeighborIndex(const unsigned int vertexIndex,
                   const unsigned int neighborNumber) const {
#ifdef DEBUG_MODE
    checkNeighborNumber(vertexIndex, neighborNumber);
#endif
    return neighborIndices_[delimiters_[vertexIndex] + neighborNumber];
  }

  const_iterator
  getNeighborhoodBegin(const unsigned int vertexIndex) const {
#ifdef DEBUG_MODE
    checkVertexIndex(vertexIndex);
#endif
    return neighborIndices_.begin() + delimiters_[vertexIndex];
  }

  const_iterator
  getNeighborhoodEnd(const unsigned int vertexIndex) const {
#ifdef DEBUG_MODE
    checkVertexIndex(vertexIndex);
#endif
    return neighborIndices_.begin() + delimiters_[vertexIndex + 1];
  }

  unsigned int
  getNumberOfVertices() const {
    return numberOfVertices_;
  }

  unsigned int
  getNumberOfEdges() const {
    return neighborIndices_.size();
  }

  // the buildCompressedGraph function will be a friend so that it can
  //  build the data of this graph.
  template <class Graph>
  friend
  CompressedGraph
  GraphUtilities::buildCompressedGraph(const Graph & graph);
};

class VectorOfVectorsGraph {

  std::vector<std::vector<unsigned int> > neighborLists_;

  void
  checkVertexIndex(const unsigned int vertexIndex) const {
    if (vertexIndex >= neighborLists_.size()) {
      throwException("invalid vertexIndex = %u for graph with "
                     "%zu vertices", vertexIndex, neighborLists_.size());
    }
  }

  void
  checkNeighborNumber(const unsigned int vertexIndex,
                      const unsigned int neighborNumber) const {
    checkVertexIndex(vertexIndex);
    const unsigned int numberOfNeighbors =
      neighborLists_[vertexIndex].size();
    if (neighborNumber >= numberOfNeighbors) {
      throwException("invalid neighborNumber = %u for vertex %u which has %u "
                     "neighbors.", neighborNumber, vertexIndex,
                     numberOfNeighbors);
    }
  }

public:

  typedef std::vector<unsigned int>::const_iterator    const_iterator;

  unsigned int
  getNumberOfNeighbors(const unsigned int vertexIndex) const {
#ifdef DEBUG_MODE
    checkVertexIndex(vertexIndex);
#endif
    return neighborLists_[vertexIndex].size();
  }

  unsigned int
  getNeighborIndex(const unsigned int vertexIndex,
                   const unsigned int neighborNumber) const {
#ifdef DEBUG_MODE
    checkNeighborNumber(vertexIndex, neighborNumber);
#endif
    return neighborLists_[vertexIndex][neighborNumber];
  }

  const_iterator
  getNeighborhoodBegin(const unsigned int vertexIndex) const {
#ifdef DEBUG_MODE
    checkVertexIndex(vertexIndex);
#endif
    return neighborLists_[vertexIndex].begin();
  }

  const_iterator
  getNeighborhoodEnd(const unsigned int vertexIndex) const {
#ifdef DEBUG_MODE
    checkVertexIndex(vertexIndex);
#endif
    return neighborLists_[vertexIndex].end();
  }

  unsigned int
  getNumberOfVertices() const {
    return neighborLists_.size();
  }

  unsigned int
  getNumberOfEdges() const {
    unsigned int numberOfEdges = 0;
    for (const std::vector<unsigned int> & neighborList : neighborLists_) {
      numberOfEdges += neighborList.size();
    }
    return numberOfEdges;
  }

  template <class RandomNumberEngine>
  friend
  VectorOfVectorsGraph
  GraphUtilities::buildRandomGraph(const unsigned int numberOfVertices,
                                   const unsigned int averageNumberOfNeighbors,
                                   const unsigned int numberOfNeighborsVariance,
                                   RandomNumberEngine * randomNumberEngine);

};

}

#endif // GRAPHS_H
