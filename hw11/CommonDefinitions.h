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

#endif // COMMON_DEFINITIONS_H
