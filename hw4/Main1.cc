// -*- C++ -*-
// Main1.cc
// cs101j 2015 hw4 Problem 1
// An example to illustrate how dynamically-sized objects and
// statically-sized objects differ in performance.

// These utilities are used on many assignments
#include "../Utilities.h"

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
// For this particular assignment, this file has the boring
//  definitions of the Point types and all the other operators that I
//  think you'll need.
#include "CommonDefinitions.h"

// Students have given feedback that they would like the functions
//  (later functors) they're supposed to work on to be split into
//  another file, so here it is.
#include "Main1_functors.h"

#include <set>

// Only bring in from the standard namespace things that we care about.
// Remember, it's a naughty thing to just use the whole namespace.
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::vector;
using std::string;
using std::array;

// As usual, a result checking function to make sure we have the right answer.
void
checkResult(const std::vector<double> & correctResult,
            const std::vector<double> & testResult,
            const std::string & testName,
            const double absoluteErrorTolerance) {
  char sprintfBuffer[500];
  if (correctResult.size() != testResult.size()) {
    sprintf(sprintfBuffer, "test result has the wrong number of entries: %zu "
            "instead of %zu, test named "
            BOLD_ON FG_RED "%s" RESET "\n",
            testResult.size(), correctResult.size(),
            testName.c_str());
    throw std::runtime_error(sprintfBuffer);
  }
  for (size_t i = 0; i < correctResult.size(); ++i) {
    const double absoluteError = std::abs(correctResult[i] - testResult[i]);
    if (absoluteError > absoluteErrorTolerance) {
      sprintf(sprintfBuffer, "wrong result for point number %zu in test result, "
              "it's %e but should be %e, test named "
              BOLD_ON FG_RED "%s" RESET "\n", i,
              testResult[i], correctResult[i], testName.c_str());
      throw std::runtime_error(sprintfBuffer);
    }
  }
}

// This is simply a sanity check on the behavior of points, to make sure
//  they all behave the same way.
template <class Point>
void
testPointClass(const std::string & pointName) {
  const Point a(1.1, 2.2, 3.3);
  const Point b(3.14, -72.1, 24.2);

  const Point c = -1.1 * a + 3.2 * b;

  const double magnitudeOfC = magnitude(c);
  const double correctMagnitude = 244.704487;
  if (std::abs(magnitudeOfC - correctMagnitude) / correctMagnitude > 1e-6) {
    char sprintfBuffer[500];
    sprintf(sprintfBuffer, "incorrect magnitude in point test: %11.4e "
            "instead of %11.4e, name " BOLD_ON FG_RED "%s" RESET "\n",
            magnitudeOfC, correctMagnitude, pointName.c_str());
    throw std::runtime_error(sprintfBuffer);
  }
}

// To emulate a real problem, we'll have containers of objects with points.
// Each object has a point, a weight, and some other junk to pretend to be a
//  real object.
template <class Point>
struct Object {
  typedef Point PointType;
  Point _position;
  double _weight;
  double _ignoredOtherJunk[10];

  Object() {
  }

  Object(const Point & position, const double weight) :
    _position(position), _weight(weight) {
  }

  Object& operator=(const Object & other) {
    _position = other._position;
    _weight = other._weight;
    // ignore junk
    return *this;
  }

};

// In this example, we'll use a std::set of objects.  However, that means that
//  the objects need to be comparable.  Here's their comparator, which uses
//  the weight, because why not?
template <class Object>
struct
ObjectSortComparator {
  bool
  operator()(const Object & a, const Object & b) {
    return a._weight < b._weight;
  }
};
template <class Object>
struct
ObjectEqualityComparator {
  bool
  operator()(const Object & a, const Object & b) {
    return a._weight == b._weight;
  }
};

// Make an object type for each type of point.
typedef Object<StaticPoint>           StaticObject;
typedef Object<DynamicPoint>          DynamicObject;
typedef Object<ArrayOfPointersPoint>  ArrayOfPointersObject;
typedef Object<VectorOfPointersPoint> VectorOfPointersObject;
typedef Object<ListPoint>             ListObject;

template <class ObjectType>
void
sortAndUniqueifyVector(std::vector<ObjectType> * vectorOfObjects) {
  std::sort(vectorOfObjects->begin(), vectorOfObjects->end(),
            ObjectSortComparator<ObjectType>());
  const auto lastUnique =
    std::unique(vectorOfObjects->begin(), vectorOfObjects->end(),
                ObjectEqualityComparator<ObjectType>());
  vectorOfObjects->erase(lastUnique, vectorOfObjects->end());
}

template <class Functor, class Iterator>
void
runTimingTest(const unsigned int numberOfTrials,
              const unsigned int numberOfNeighbors,
              const Functor functor,
              const Iterator beginIterator,
              const Iterator endIterator,
              std::vector<double> * result,
              double * elapsedTime) {

  *elapsedTime = std::numeric_limits<double>::max();

  for (unsigned int trialNumber = 0;
       trialNumber < numberOfTrials; ++trialNumber) {

    result->resize(0);

    // Note: we intentionally don't clear the cache.  We're not looking
    //  for caching effects, we're looking for how the calculation works
    //  in the *best* of conditions.

    // Start timing
    const high_resolution_clock::time_point tic =
      high_resolution_clock::now();

    // Do the calculation
    functor.calculateNeighborhoodLocality(numberOfNeighbors,
                                          beginIterator,
                                          endIterator,
                                          std::back_inserter(*result));

    // Stop timing
    high_resolution_clock::time_point toc =
      high_resolution_clock::now();
    *elapsedTime =
      std::min(*elapsedTime,
               duration_cast<duration<double> >(std::chrono::operator-(toc, tic)).count());
  }

}

int main() {

  char sprintfBuffer[500];

  // Perform a quick test of the points.
  printf("testing point classes\n");
  testPointClass<StaticPoint>("StaticPoint");
  testPointClass<DynamicPoint>("DynamicPoint");
  testPointClass<ArrayOfPointersPoint>("ArrayOfPointersPoint");
  testPointClass<VectorOfPointersPoint>("VectorOfPointersPoint");
  testPointClass<ListPoint>("ListPoint");
  printf("point tests passed\n");

  // ===========================================================================
  // *************************** < Inputs> *************************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  const unsigned int numberOfObjects                    = 1e4;
  const array<unsigned int, 2> rangeOfNumberOfNeighbors = {{1, 100}};
  const unsigned int numberOfDataPoints                 = 15;
  const unsigned int numberOfTrialsForExpensiveMethods  = 2;
  const unsigned int numberOfTrialsForCheapMethods      = 2;

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // *************************** </Inputs> *************************************
  // ===========================================================================

  // On each test, we need to make sure we get the same result.  A test will
  //  fail if the difference between any entry in our result is more than
  //  absoluteErrorTolerance different than entries we got with another method.
  const double absoluteErrorTolerance = 1e-4;

  // This prefix and suffix will determine where files will be written and
  //  their names.
  const string prefix = "data/Main1_";
  const string suffix = "_shuffler";

  // Make sure that the data directory exists.
  Utilities::verifyThatDirectoryExists("data");

  // Make a random number generator
  std::default_random_engine randomNumberEngine;
  std::uniform_real_distribution<double> randomNumberGenerator(0, 1);

  // Create the containers.
  // Vectors
  std::vector<StaticObject> vectorOfStaticObjects(numberOfObjects);
  std::vector<DynamicObject> vectorOfDynamicObjects(numberOfObjects);
  std::vector<ArrayOfPointersObject> vectorOfArrayOfPointersObjects(numberOfObjects);
  std::vector<VectorOfPointersObject> vectorOfVectorOfPointersObjects(numberOfObjects);
  std::vector<ListObject> vectorOfListObjects(numberOfObjects);

  // Sets
  std::set<StaticObject, ObjectSortComparator<StaticObject> > setOfStaticObjects;
  std::set<DynamicObject, ObjectSortComparator<DynamicObject> > setOfDynamicObjects;
  std::set<ArrayOfPointersObject,
           ObjectSortComparator<ArrayOfPointersObject> > setOfArrayOfPointersObjects;
  std::set<VectorOfPointersObject,
           ObjectSortComparator<VectorOfPointersObject> > setOfVectorOfPointersObjects;
  std::set<ListObject, ObjectSortComparator<ListObject> > setOfListObjects;
  // Populate the containers
  for (unsigned int objectIndex = 0;
       objectIndex < numberOfObjects; ++objectIndex) {
    const double weight = randomNumberGenerator(randomNumberEngine);

    const StaticPoint staticPoint(randomNumberGenerator(randomNumberEngine),
                                  randomNumberGenerator(randomNumberEngine),
                                  randomNumberGenerator(randomNumberEngine));
    const StaticObject staticObject(staticPoint, weight);
    vectorOfStaticObjects[objectIndex] = staticObject;
    setOfStaticObjects.insert(staticObject);

    const DynamicPoint dynamicPoint(staticPoint[0],
                                    staticPoint[1],
                                    staticPoint[2]);
    const DynamicObject dynamicObject(dynamicPoint, weight);
    vectorOfDynamicObjects[objectIndex] = dynamicObject;
    setOfDynamicObjects.insert(dynamicObject);

    const ArrayOfPointersPoint arrayOfPointersPoint(staticPoint[0],
                                                    staticPoint[1],
                                                    staticPoint[2]);
    const ArrayOfPointersObject arrayOfPointersObject(arrayOfPointersPoint,
                                                      weight);
    vectorOfArrayOfPointersObjects[objectIndex] = arrayOfPointersObject;
    setOfArrayOfPointersObjects.insert(arrayOfPointersObject);

    const VectorOfPointersPoint vectorOfPointersPoint(staticPoint[0],
                                                      staticPoint[1],
                                                      staticPoint[2]);
    const VectorOfPointersObject vectorOfPointersObject(vectorOfPointersPoint,
                                                        weight);
    vectorOfVectorOfPointersObjects[objectIndex] = vectorOfPointersObject;
    setOfVectorOfPointersObjects.insert(vectorOfPointersObject);

    const ListPoint listPoint(staticPoint[0],
                              staticPoint[1],
                              staticPoint[2]);
    const ListObject listObject(listPoint, weight);
    vectorOfListObjects[objectIndex] = listObject;
    setOfListObjects.insert(listObject);
  }

  // Sort the vector into the same order as the set, so we get the same
  //  answers.  Then, remove repeats.
  sortAndUniqueifyVector<StaticObject>(&vectorOfStaticObjects);
  sortAndUniqueifyVector<DynamicObject>(&vectorOfDynamicObjects);
  sortAndUniqueifyVector<ArrayOfPointersObject>(&vectorOfArrayOfPointersObjects);
  sortAndUniqueifyVector<VectorOfPointersObject>(&vectorOfVectorOfPointersObjects);
  sortAndUniqueifyVector<ListObject>(&vectorOfListObjects);
  // Make sure it worked correctly
  if (vectorOfStaticObjects.size() != setOfStaticObjects.size()) {
    sprintf(sprintfBuffer, "could not get the set and vector to be of the "
            "same length: vector is %zu, set is %zu\n",
            vectorOfStaticObjects.size(), setOfStaticObjects.size());
    throw std::runtime_error(sprintfBuffer);
  }

  // Open the file that will store the data.
  sprintf(sprintfBuffer, "%sdata%s.csv", prefix.c_str(), suffix.c_str());
  FILE* file = fopen(sprintfBuffer, "w");

  // Write out the csv headers for our three techniques.
  fprintf(file, "numberOfNeighbors"
          ", Vector_StaticPoint, Vector_DynamicPoint"
          ", Vector_ArrayOfPointersPoint, Vector_VectorOfPointersPoint"
          ", Vector_ListPoint"
          ", Set_StaticPoint, Set_DynamicPoint"
          ", Set_ArrayOfPointersPoint, Set_VectorOfPointersPoint"
          ", Set_ListPoint, "
          ", Vector_StaticPoint_Improved"
          ", Set_VectorOfPointersPoint_Improved\n");

  // For each number of neighbors
  for (unsigned int dataPointIndex = 0;
       dataPointIndex < numberOfDataPoints;
       ++dataPointIndex) {
    // Get this number of neighbors
    const unsigned int numberOfNeighbors =
      Utilities::interpolateNumberLinearlyOnLinearScale(rangeOfNumberOfNeighbors[0],
                                                        rangeOfNumberOfNeighbors[1],
                                                        numberOfDataPoints,
                                                        dataPointIndex);

    fprintf(file, "%4u", numberOfNeighbors);

    const high_resolution_clock::time_point thisSizesTic =
      high_resolution_clock::now();

    // Results from each test
    std::vector<double> result;
    result.reserve(numberOfObjects);
    double elapsedTime;

    // Run the test
    runTimingTest(numberOfTrialsForCheapMethods,
                  numberOfNeighbors,
                  Main1::LocalityCalculator(),
                  vectorOfStaticObjects.begin(),
                  vectorOfStaticObjects.end(),
                  &result,
                  &elapsedTime);
    fprintf(file, ", %10.6e", elapsedTime);

    // Store the "right" answer
    const std::vector<double> correctResult = result;

    // Run the test
    runTimingTest(numberOfTrialsForExpensiveMethods,
                  numberOfNeighbors,
                  Main1::LocalityCalculator(),
                  vectorOfDynamicObjects.begin(),
                  vectorOfDynamicObjects.end(),
                  &result,
                  &elapsedTime);
    // Check the result
    checkResult(correctResult, result,
                std::string("vector of DynamicObject"),
                absoluteErrorTolerance);
    fprintf(file, ", %10.6e", elapsedTime);

    // Run the test
    runTimingTest(numberOfTrialsForExpensiveMethods,
                  numberOfNeighbors,
                  Main1::LocalityCalculator(),
                  vectorOfArrayOfPointersObjects.begin(),
                  vectorOfArrayOfPointersObjects.end(),
                  &result,
                  &elapsedTime);
    // Check the result
    checkResult(correctResult, result,
                std::string("vector of ArrayOfPointersObject"),
                absoluteErrorTolerance);
    fprintf(file, ", %10.6e", elapsedTime);

    // Run the test
    runTimingTest(numberOfTrialsForExpensiveMethods,
                  numberOfNeighbors,
                  Main1::LocalityCalculator(),
                  vectorOfVectorOfPointersObjects.begin(),
                  vectorOfVectorOfPointersObjects.end(),
                  &result,
                  &elapsedTime);
    // Check the result
    checkResult(correctResult, result,
                std::string("vector of VectorOfPointersObject"),
                absoluteErrorTolerance);
    fprintf(file, ", %10.6e", elapsedTime);

    // Run the test
    runTimingTest(numberOfTrialsForExpensiveMethods,
                  numberOfNeighbors,
                  Main1::LocalityCalculator(),
                  vectorOfListObjects.begin(),
                  vectorOfListObjects.end(),
                  &result,
                  &elapsedTime);
    // Check the result
    checkResult(correctResult, result,
                std::string("vector of ListObject"),
                absoluteErrorTolerance);
    fprintf(file, ", %10.6e", elapsedTime);

    // Run the test
    runTimingTest(numberOfTrialsForExpensiveMethods,
                  numberOfNeighbors,
                  Main1::LocalityCalculator(),
                  setOfStaticObjects.begin(),
                  setOfStaticObjects.end(),
                  &result,
                  &elapsedTime);
    // Check the result
    checkResult(correctResult, result,
                std::string("set of StaticObject"),
                absoluteErrorTolerance);
    fprintf(file, ", %10.6e", elapsedTime);

    // Run the test
    runTimingTest(numberOfTrialsForExpensiveMethods,
                  numberOfNeighbors,
                  Main1::LocalityCalculator(),
                  setOfDynamicObjects.begin(),
                  setOfDynamicObjects.end(),
                  &result,
                  &elapsedTime);
    // Check the result
    checkResult(correctResult, result,
                std::string("set of DynamicObject"),
                absoluteErrorTolerance);
    fprintf(file, ", %10.6e", elapsedTime);

    // Run the test
    runTimingTest(numberOfTrialsForExpensiveMethods,
                  numberOfNeighbors,
                  Main1::LocalityCalculator(),
                  setOfArrayOfPointersObjects.begin(),
                  setOfArrayOfPointersObjects.end(),
                  &result,
                  &elapsedTime);
    // Check the result
    checkResult(correctResult, result,
                std::string("set of ArrayOfPointersObject"),
                absoluteErrorTolerance);
    fprintf(file, ", %10.6e", elapsedTime);

    // Run the test
    runTimingTest(numberOfTrialsForExpensiveMethods,
                  numberOfNeighbors,
                  Main1::LocalityCalculator(),
                  setOfVectorOfPointersObjects.begin(),
                  setOfVectorOfPointersObjects.end(),
                  &result,
                  &elapsedTime);
    // Check the result
    checkResult(correctResult, result,
                std::string("set of VectorOfPointersObject"),
                absoluteErrorTolerance);
    fprintf(file, ", %10.6e", elapsedTime);

    // Run the test
    runTimingTest(numberOfTrialsForExpensiveMethods,
                  numberOfNeighbors,
                  Main1::LocalityCalculator(),
                  setOfListObjects.begin(),
                  setOfListObjects.end(),
                  &result,
                  &elapsedTime);
    // Check the result
    checkResult(correctResult, result,
                std::string("set of ListObject"),
                absoluteErrorTolerance);
    fprintf(file, ", %10.6e", elapsedTime);

    // Run the test
    runTimingTest(numberOfTrialsForCheapMethods,
                  numberOfNeighbors,
                  Main1::ImprovedLocalityCalculator(),
                  vectorOfStaticObjects.begin(),
                  vectorOfStaticObjects.end(),
                  &result,
                  &elapsedTime);
    // Check the result
    checkResult(correctResult, result,
                std::string("vector of StaticObject improved"),
                absoluteErrorTolerance);
    fprintf(file, ", %10.6e", elapsedTime);

    // Run the test
    runTimingTest(numberOfTrialsForCheapMethods,
                  numberOfNeighbors,
                  Main1::ImprovedLocalityCalculator(),
                  setOfVectorOfPointersObjects.begin(),
                  setOfVectorOfPointersObjects.end(),
                  &result,
                  &elapsedTime);
    // Check the result
    checkResult(correctResult, result,
                std::string("set of VectorOfPointersObject improved"),
                absoluteErrorTolerance);
    fprintf(file, ", %10.6e", elapsedTime);

    // Output heartbeat message
    const high_resolution_clock::time_point thisSizesToc =
      high_resolution_clock::now();
    const double thisSizesElapsedTime =
      duration_cast<duration<double> >(std::chrono::operator-(thisSizesToc, thisSizesTic)).count();
    printf("processing %4u neighbors took %7.2f seconds\n",
           numberOfNeighbors,
           thisSizesElapsedTime);

    fprintf(file, "\n");
    fflush(file);
  }
  fclose(file);

  return 0;
}
