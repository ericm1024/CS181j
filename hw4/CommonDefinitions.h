// -*- C++ -*-
#ifndef COMMON_DEFINITIONS_H
#define COMMON_DEFINITIONS_H

// c junk
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdio>

// c++ junk
#include <array>
#include <vector>
#include <list>
#include <string>
#include <algorithm>
#include <chrono>
#include <random>
#include <fstream>

// Ths is an implementation of a point which uses a std::array as the
//  underlying data structure.
struct StaticPoint {

  // The default constructor's job is to essentially just make room
  //  for the data that will be stored.  For this class, it doesn't
  //  have to do anything because the data is stored in a std::array.
  StaticPoint() {
  }

  // This is the copy constructor.  Its job is to allocate data for
  //  this instance of the class, then "make this instance of the
  //  class be the same as the other instance of the class".
  StaticPoint(const StaticPoint & other) {
    for (unsigned int i = 0; i < 3; ++i) {
      _data[i] = other._data[i];
    }
  }

  // This is the argumented or parameterized construtor.  Its job is
  //  to allocate room for this instance's data, then store the
  //  provided input values.
  StaticPoint(const double x, const double y, const double z) {
    _data[0] = x;
    _data[1] = y;
    _data[2] = z;
  }

  // This is the assignment operator.  It does NOT allocate data.  Its
  //  job is to "make this instance of the class be the same as the
  //  other instance of the class".
  StaticPoint& operator=(const StaticPoint & other) {
    for (unsigned int i = 0; i < 3; ++i) {
      _data[i] = other._data[i];
    }
    return *this;
  }

  // This is the bracket operator, specifically the mutator.  It will
  //  be given an index from 0 to 2 and its job is to return that
  //  coordinate of the point.  It returns by nonconst reference
  //  because this is how people set values in the point.
  double &
  operator[](const unsigned int index) {
    return _data[index];
  }

  // This is the bracket operator, specifically the accessor.  It will
  //  be given an index from 0 to 2 and its job is to return that
  //  coordinate of the point.  It returns by value because this is
  //  how people just read values in the point.
  double
  operator[](const unsigned int index) const {
    return _data[index];
  }

  // This is the destructor, whose job it is to release any memory
  //  that was allocated by the constructor that was called.  Because
  //  the underlying data for this class is simply a std::array, this
  //  doesn't have to do anything.
  ~StaticPoint() {
    // nothing to do!
  }

private:
  std::array<double, 3> _data;
};





struct DynamicPoint {

  DynamicPoint() {
    // TODO?
  }

  DynamicPoint(const DynamicPoint & other) {
    // TODO?
  }

  DynamicPoint(const double x, const double y, const double z) {
    // TODO?
  }

  DynamicPoint& operator=(const DynamicPoint & other) {
    // TODO?
    return *this;
  }

  double &
  operator[](const unsigned int index) {
    // TODO?
  }

  double
  operator[](const unsigned int index) const {
    // TODO?
  }

  ~DynamicPoint() {
    // TODO?
  }

private:
  std::vector<double> _data;
};






struct ArrayOfPointersPoint {

  ArrayOfPointersPoint() {
    // TODO?
  }

  ArrayOfPointersPoint(const ArrayOfPointersPoint & other) {
    // TODO?
  }

  ArrayOfPointersPoint(const double x, const double y, const double z) {
    // TODO?
  }

  ArrayOfPointersPoint& operator=(const ArrayOfPointersPoint & other) {
    // TODO?
    return *this;
  }

  double &
  operator[](const unsigned int index) {
    // TODO?
  }

  double
  operator[](const unsigned int index) const {
    // TODO?
  }

  ~ArrayOfPointersPoint() {
    // TODO?
  }

private:
  std::array<double*, 3> _data;
};






struct VectorOfPointersPoint {

  VectorOfPointersPoint() : _data(3) {
    // TODO?
  }

  VectorOfPointersPoint(const VectorOfPointersPoint & other) : _data(3) {
    // TODO?
  }

  VectorOfPointersPoint(const double x, const double y, const double z) {
    // TODO?
  }

  VectorOfPointersPoint& operator=(const VectorOfPointersPoint & other) {
    // TODO?
    return *this;
  }

  double &
  operator[](const unsigned int index) {
    // TODO?
  }

  double
  operator[](const unsigned int index) const {
    // TODO?
  }

  ~VectorOfPointersPoint() {
    // TODO?
  }

private:
  std::vector<double*> _data;
};







struct ListPoint {

  ListPoint() {
    // TODO?
  }

  ListPoint(const ListPoint & other) {
    // TODO?
  }

  ListPoint(const double x, const double y, const double z) {
    // TODO?
  }

  ListPoint& operator=(const ListPoint & other) {
    // TODO?
    return *this;
  }

  double &
  operator[](const unsigned int index) {
    // TODO?
  }

  double
  operator[](const unsigned int index) const {
    // TODO?
  }

  ~ListPoint() {
  }

private:
  std::list<double> _data;
};












// some general point operations

template <class Point>
Point operator-(const Point & x, const Point & y ) {
  return Point(x[0] - y[0], x[1] - y[1], x[2] - y[2]);
}

template <class Point>
Point operator+(const Point & x, const Point & y ) {
  return Point(x[0] + y[0], x[1] + y[1], x[2] + y[2]);
}

template <class Point>
Point operator*(const Point & x, const double d ) {
  return Point(x[0]*d, x[1]*d, x[2]*d);
}

template <class Point>
Point operator*(const double d, const Point & x) {
  return operator*(x, d);
}

template <class Point>
Point operator/(const Point & x, const double d ) {
  return operator*(x, 1/d);
}

template <class Point>
Point operator/(const double d, const Point & x) {
  return operator*(x, 1/d);
}

template <class Point>
double
magnitude(const Point & p) {
  return std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
}

#endif // COMMON_DEFINITIONS_H
