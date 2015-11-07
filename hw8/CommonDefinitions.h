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

// we use an enum so that we don't have to use magic numbers to pass the
//  style, like "1 happens to mean Random, of course!"
enum PolynomialOrderStyle {PolynomialOrderFixed,
                           PolynomialOrderProportionalToIndex};

std::string
convertPolynomialOrderStyleToString(const PolynomialOrderStyle style) {
  switch (style) {
  case PolynomialOrderFixed:
    return std::string("Fixed");
    break;
  case PolynomialOrderProportionalToIndex:
    return std::string("ProportionalToIndex");
    break;
  default:
    fprintf(stderr, "unimplemented polynomial order style\n");
    exit(1);
  }
}

typedef std::array<double, 3> Point;

Point operator-(const Point & x, const Point & y ) {
  return (Point) {{x[0] - y[0], x[1] - y[1], x[2] - y[2]}};
}

Point operator+(const Point & x, const Point & y ) {
  return (Point) {{x[0] + y[0], x[1] + y[1], x[2] + y[2]}};
}

Point operator*(const Point & x, const double d ) {
  return (Point) {{x[0]*d, x[1]*d, x[2]*d}};
}

Point operator*(const double d, const Point & x) {
  return operator*(x, d);
}

Point operator/(const Point & x, const double d ) {
  return operator*(x, 1/d);
}

Point operator/(const double d, const Point & x) {
  return operator*(x, 1/d);
}

Point & operator+=(Point & x, const Point & y ) {
  x[0] += y[0];
  x[1] += y[1];
  x[2] += y[2];
  return x;
}

double
magnitude(const Point & p) {
  return std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
}

double
squaredMagnitude(const Point & p) {
  return p[0]*p[0] + p[1]*p[1] + p[2]*p[2];
}

#endif // COMMON_DEFINITIONS_H
