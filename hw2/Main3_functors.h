// -*- C++ -*-
#ifndef MAIN3_FUNCTORS_H
#define MAIN3_FUNCTORS_H

#include <vector>
#include <map>

// This is a simple power calculating function, which represents the
//  work done for each input.
// Being able to tune the amount of work will help us explore the
//  tradeoffs between memoization and recomputation depending on the
//  cost of recomputation.
inline
double calculatePower(const double x, const unsigned int power) {
  double y = 1;
  for (unsigned int powerIndex = 0; powerIndex < power; ++powerIndex) {
    y *= x;
  }
  return y;
}

// A PowerCalculator object is used to compute powers of doubles.
// Once initialized with a power, PowerCalculator.computePowers(..)
//  computes that power of all of the elements in the input.
struct
PowerCalculator {
  const unsigned int _power;

  // Constructs a PowerCalculator for exponent power.
  PowerCalculator(const unsigned int power) :
    _power(power) {
    // Nothing to do here.
  }

  // Sets each element in result to _power of the corresponding element
  //  in input.
  void computePowers(const std::vector<double> & input,
                     std::vector<double> * result) {
    const unsigned int inputSize = input.size();
    for (unsigned int i = 0; i < inputSize; ++i) {
      result->at(i) = calculatePower(input[i], _power);
    }
  }
};

// When we memoize, we will settle for the result of any previous
//  input that is "close enough" to our current input. This comparator
//  encodes "close enough" in a way that std::map can understand. This
//  way, if you ask for the power of .00001 and then the power of
//  .000011, the PowerCalculator can use the result from .00001 as a
//  good approximation of the latter power.
// STUDENTS: YOU DO NOT NEED TO MODIFY THIS STRUCT.
template <class T>
struct TolerancedFloatCompare {
  T _tolerance;
  // Use the square root of epsilon for the default tolerance, i.e. use half
  // of the significant digits.
  TolerancedFloatCompare(const T tolerance =
                         std::sqrt(std::numeric_limits<T>::epsilon())) :
    _tolerance(tolerance) {
  }
  bool operator()(T a, T b) const {
    return (std::abs(a - b) > _tolerance) && (a < b);
  }
};

// A MapMemoizedPowerCalculator object is used to compute powers of
//  doubles. However, rather than recomputing for every input, this
//  version memoizes all results in a std::map (tree). Hopefully this
//  will save us a lot of flops! /s
struct
MapMemoizedPowerCalculator {
  const unsigned int _power;
  const double _memoizationResolution;
  std::map<double, double, TolerancedFloatCompare<double> > _storage;

  MapMemoizedPowerCalculator(const unsigned int power,
                             const double memoizationResolution) :
    _power(power),
    _memoizationResolution(memoizationResolution),
    _storage(TolerancedFloatCompare<double>(_memoizationResolution)) {
    // Nothing to do here.
  }

  // Sets each element in result to _power of the corresponding
  //  element in input.
  void computePowers(const std::vector<double> & input,
                     std::vector<double> * result) {
    const unsigned int inputSize = input.size();
    for (unsigned int i = 0; i < inputSize; ++i) {
      const double x = input[i];
      // If we've already memoized x (or something close to it), just
      //  return the result!
      const auto it = _storage.find(x);
      if (it != _storage.end()) {
        result->at(i) = it->second;
      } else {
        // We didn't find it, we'd better compute and store
        const double y = calculatePower(input[i], _power);
        _storage.insert(std::make_pair(x, y));
        result->at(i) = y;
      }
    }
  }
};

// An ArrayMemoizedPowerCalculator object is used to compute powers of
//  doubles. However, rather than recomputing for every input, this
//  version memoizes all results in an array. Since all inputs are in
//  the range 0 through 1 and you want to memoize inputs that are
//  within memoizationResolution of each other, you'll need something
//  around 1/memoizationResolution buckets in your array.
// STUDENTS: THIS IS THE STRUCT YOU HAVE TO MODIFY.
struct
ArrayMemoizedPowerCalculator {
  const unsigned int _power;
  const double _memoizationResolution;
  std::vector<double> _storage;

  ArrayMemoizedPowerCalculator(const unsigned int power,
                               const double memoizationResolution) :
    _power(power),
    _memoizationResolution(memoizationResolution) {
    // TODO: initialize _storage
  }

  // Sets each element in result to _power of the corresponding element
  //  in input.
  void computePowers(const std::vector<double> & input,
                     std::vector<double> * result) {

    throw std::runtime_error("You haven't yet implemented the array "
                             "memoized version!");

    // TODO: use array memoization

  }
};

#endif // MAIN3_FUNCTORS_H
