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

#include <array>

char exceptionBuffer[10000];
#define throwException(s, ...)                                  \
  sprintf(exceptionBuffer, "%s:%s:%d: " s, __FILE__, __func__,  \
          __LINE__, ##__VA_ARGS__);                             \
  throw std::runtime_error(exceptionBuffer);

typedef std::array<float, 3> Point;

Point operator-(const Point & x, const Point & y ) {
  return (Point) {{x[0] - y[0], x[1] - y[1], x[2] - y[2]}};
}

Point operator+(const Point & x, const Point & y ) {
  return (Point) {{x[0] + y[0], x[1] + y[1], x[2] + y[2]}};
}

Point operator*(const Point & x, const float d ) {
  return (Point) {{x[0]*d, x[1]*d, x[2]*d}};
}

Point operator*(const float d, const Point & x) {
  return operator*(x, d);
}

Point operator/(const Point & x, const float d ) {
  return operator*(x, 1/d);
}

Point operator/(const float d, const Point & x) {
  return operator*(x, 1/d);
}

Point & operator+=(Point & x, const Point & y ) {
  x[0] += y[0];
  x[1] += y[1];
  x[2] += y[2];
  return x;
}

float
magnitude(const Point & p) {
  return std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
}

float
squaredMagnitude(const Point & p) {
  return p[0]*p[0] + p[1]*p[1] + p[2]*p[2];
}
#endif // COMMON_DEFINITIONS_H
