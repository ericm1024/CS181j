// -*- C++ -*-
#ifndef MAIN1_FUNCTORS_H
#define MAIN1_FUNCTORS_H

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

namespace Main1 {

struct LocalityCalculator {
  template <class InputIterator, class OutputIterator>
  static
  void
  calculateNeighborhoodLocality(const unsigned int numberOfNeighbors,
                                const InputIterator begin,
                                const InputIterator end,
                                OutputIterator outputIterator) {
    // This is voodoo magic to extract the point type out of the iterator.
    // Don't get stuck on this, this is just keeping us from templating on both
    //  the iterator and the point type, because they're really related.
    typedef typename std::iterator_traits<InputIterator>::value_type ObjectType;
    typedef typename ObjectType::PointType                           Point;

    // Form neighborhood iterators
    InputIterator neighborhoodBeginIterator = begin;
    InputIterator neighborhoodEndIterator = begin;
    std::advance(neighborhoodEndIterator, 2 * numberOfNeighbors + 1);
    InputIterator currentObjectIterator = begin;
    std::advance(currentObjectIterator, numberOfNeighbors);

    // For each object, making sure neighborhood isn't reaching past end
    while(neighborhoodEndIterator != end) {
      // Grab this object's point
      const Point & thisPoint = currentObjectIterator->_position;
      // Calculate distances to each neighbor, sum
      Point localCentroid(0., 0., 0.);
      // Iterate over neighborhood
      for (InputIterator neighborhoodObjectIterator = neighborhoodBeginIterator;
           neighborhoodObjectIterator != neighborhoodEndIterator;
           ++neighborhoodObjectIterator) {
        localCentroid = localCentroid +
          neighborhoodObjectIterator->_position *
          neighborhoodObjectIterator->_weight;
      }
      localCentroid = localCentroid / (2 * numberOfNeighbors + 1);
      const double thisObjectsNeighborhoodLocality =
        magnitude(localCentroid - thisPoint);

      *outputIterator = thisObjectsNeighborhoodLocality;

      // Increment current object, neighborhood bounds, and output
      ++currentObjectIterator;
      ++neighborhoodBeginIterator;
      ++neighborhoodEndIterator;
      ++outputIterator;
    }
  }
};

struct ImprovedLocalityCalculator {
  template <class InputIterator, class OutputIterator>
  static
  void
  calculateNeighborhoodLocality(const unsigned int numberOfNeighbors,
                                const InputIterator begin,
                                const InputIterator end,
                                OutputIterator outputIterator) {
    // This is voodoo magic to extract the point type out of the iterator.
    // Don't get stuck on this, this is just keeping us from templating on both
    //  the iterator and the point type, because they're really related.
    typedef typename std::iterator_traits<InputIterator>::value_type ObjectType;
    typedef typename ObjectType::PointType                           Point;

    // TODO: change me somehow

    // Form neighborhood iterators
    InputIterator neighborhoodBeginIterator = begin;
    InputIterator neighborhoodEndIterator = begin;
    std::advance(neighborhoodEndIterator, 2 * numberOfNeighbors + 1);
    InputIterator currentObjectIterator = begin;
    std::advance(currentObjectIterator, numberOfNeighbors);

    // For each object, making sure neighborhood isn't reaching past end
    while(neighborhoodEndIterator != end) {
      // Grab this object's point
      const Point & thisPoint = currentObjectIterator->_position;
      // Calculate distances to each neighbor, sum
      Point localCentroid(0., 0., 0.);
      // Iterate over neighborhood
      for (InputIterator neighborhoodObjectIterator = neighborhoodBeginIterator;
           neighborhoodObjectIterator != neighborhoodEndIterator;
           ++neighborhoodObjectIterator) {
        localCentroid = localCentroid +
          neighborhoodObjectIterator->_position *
          neighborhoodObjectIterator->_weight;
      }
      localCentroid = localCentroid / (2 * numberOfNeighbors + 1);
      const double thisObjectsNeighborhoodLocality =
        magnitude(localCentroid - thisPoint);

      *outputIterator = thisObjectsNeighborhoodLocality;

      // Increment current object, neighborhood bounds, and output
      ++currentObjectIterator;
      ++neighborhoodBeginIterator;
      ++neighborhoodEndIterator;
      ++outputIterator;
    }
  }
};

}

#endif // MAIN1_FUNCTORS_H
