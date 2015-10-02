// -*- C++ -*-
// Main2.cc
// cs181j hw4 problem 2
// This is a test case examining how many times the copy constructor and
//  malloc can be called.

// c junk
#include <cstdio>
#include <cstdlib>
#include <cmath>

// c++ junk
#include <vector>

// We can keep track of how many times malloc has been called with this
//  little utility library, which I've modified a bit.
#include "malloc_count-0.7/malloc_count.h"

// In order to keep track of the number of copy constructors, I use a
//  global variable here.  This is nasty, evil, and wrong.  Let's keep this
//  our little secret.
unsigned int numberOfCopyConstructors = 0;

// The Object simply keeps track of the number of copy constructors
struct Object {
  Object() {
  }
  Object(const Object & o) {
    ++numberOfCopyConstructors;
  }
};

// This class is a thin wrapping around std::vector.  I have to do
//  this for our malloc predictions to work correctly.  In class,
//  we'll talk about why we can't just use std::vector.
struct WrappedVector {
  std::vector<double> _v;
  WrappedVector() : _v() {
  }
  WrappedVector(const unsigned int size) : _v(size) {
  }
  WrappedVector(const WrappedVector & other) : _v(other._v) {
  }
  void
  push_back(const double d) {
    _v.push_back(d);
  }
  double &
  operator[](const unsigned int index) {
    return _v[index];
  }
  const double
  operator[](const unsigned int index) const {
    return _v[index];
  }
  void
  clear() {
    _v.clear();
  }
};

int main(int argc, char ** argv) {

  // Here, we try to predict the number of copy constructors called when
  //  pushing objects back onto a vector of Objects.
  for (unsigned int numberOfObjects = 2; numberOfObjects < 17; ++numberOfObjects) {

    // Reset the counter
    numberOfCopyConstructors = 0;

    // Perform the test
    std::vector<Object> objects;
    for (unsigned int i = 0; i < numberOfObjects; ++i) {
      objects.push_back(Object());
    }

    // Make a prediction
    const unsigned int predictedNumberOfCopyConstructors = 0; // TODO
    printf("%2u objects, predicted %3u copy constructors, observed %3u\n",
           numberOfObjects, predictedNumberOfCopyConstructors,
           numberOfCopyConstructors);
    // Check the prediction
    if (predictedNumberOfCopyConstructors != numberOfCopyConstructors) {
      printf("error in prediction\n");
      exit(1);
    }
  }




  // Here, we try to predict the number of mallocs performed when
  //  pushing objects back onto a vector of Objects.
  for (unsigned int numberOfObjects = 2; numberOfObjects < 17; ++numberOfObjects) {

    // Reset the counter
    resetNumberOfTimesMallocHasBeenCalled();

    // Perform the test
    std::vector<Object> objects;
    for (unsigned int i = 0; i < numberOfObjects; ++i) {
      objects.push_back(Object());
    }

    // Get the value of the counter
    const unsigned int numberOfMallocs = getNumberOfTimesMallocHasBeenCalled();
    // Make a prediction
    const unsigned int predictedNumberOfMallocs = 0; // TODO
    printf("%2u objects, predicted %4u mallocs, observed %4u\n",
           numberOfObjects,
           predictedNumberOfMallocs,
           numberOfMallocs);
    // Check the prediction
    if (predictedNumberOfMallocs != numberOfMallocs) {
      printf("error in prediction\n");
      exit(1);
    }
  }




  // Here, we try to predict the number of mallocs performed when
  //  pushing objects that require mallocs onto a vector of those objects.
  const std::vector<unsigned int> initialNumbersOfWrappedVectors = {3, 17, 36};
  for (const unsigned int initialNumberOfWrappedVectors :
         initialNumbersOfWrappedVectors) {

    // Reset the counter
    resetNumberOfTimesMallocHasBeenCalled();

    // Perform the test
    std::vector<WrappedVector> wrappedVectors(initialNumberOfWrappedVectors);
    for (unsigned int i = 0; i < initialNumberOfWrappedVectors; ++i) {
      wrappedVectors[i].push_back(i);
      wrappedVectors[i].push_back(i);
      wrappedVectors[i].push_back(i);
      wrappedVectors[i].push_back(i);
    }
    wrappedVectors.reserve(initialNumberOfWrappedVectors + 1);

    // Get the value of the counter
    const unsigned int numberOfMallocs = getNumberOfTimesMallocHasBeenCalled();
    // Make a prediction
    const unsigned int predictedNumberOfMallocs = 0; // TODO

    printf("predicted %4u mallocs, observed %4u\n",
           predictedNumberOfMallocs,
           numberOfMallocs);
    // Check the prediction
    if (predictedNumberOfMallocs != numberOfMallocs) {
      printf("error in prediction\n");
      exit(1);
    }
  }




  // Here, we put it all together and predict the number of mallocs
  //  performed in a more complex nested loop.
  const std::vector<unsigned int> numbersOfOuterLoops = {15, 64, 257};
  const std::vector<unsigned int> numbersOfInnerLoops = {3, 8, 17};
  for (const unsigned int numberOfOuterLoops : numbersOfOuterLoops) {
    for (const unsigned int numberOfInnerLoops : numbersOfInnerLoops) {

      // Reset the counter
      resetNumberOfTimesMallocHasBeenCalled();

      // Perform the test
      typedef WrappedVector InnerVector;
      std::vector<InnerVector> outerVectorOfVectors;
      for (unsigned int outerLoopIndex = 0;
           outerLoopIndex < numberOfOuterLoops; ++outerLoopIndex) {
        InnerVector innerVector;
        for (unsigned int innerLoopIndex = 0;
             innerLoopIndex < numberOfInnerLoops; ++innerLoopIndex) {
          innerVector.push_back(0);
        }
        outerVectorOfVectors.push_back(innerVector);
      }

      // Get the value of the counter
      const unsigned int numberOfMallocs = getNumberOfTimesMallocHasBeenCalled();
      // Make a prediction
      const unsigned int predictedNumberOfMallocs = 0; // TODO
      printf("%3u outer, %2u inner, predicted %4u mallocs, observed %4u\n",
             numberOfOuterLoops, numberOfInnerLoops,
             predictedNumberOfMallocs,
             numberOfMallocs);
      // Check the prediction
      if (predictedNumberOfMallocs != numberOfMallocs) {
        printf("error in prediction\n");
        exit(1);
      }
    }
  }

  return 0;
}
