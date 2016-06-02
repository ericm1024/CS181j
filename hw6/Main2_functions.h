// -*- C++ -*-
#ifndef MAIN2_FUNCTIONS_H
#define MAIN2_FUNCTIONS_H

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

// These utilities are used on many assignments
#include "../Utilities.h"

// header file for openmp
#include <omp.h>

// header file for c++11 threads
#include <thread>

#include <mutex>
#include <iostream>
#include <functional>

double
calculateIntegral_serial(const unsigned int numberOfThreads,
                         const unsigned int numberOfIntervals,
                         const std::array<double, 2> & integrationRange) {
  ignoreUnusedVariable(numberOfThreads);
  double integral = 0;

  const double dx =
    (integrationRange[1] - integrationRange[0]) / numberOfIntervals;
  const double base = integrationRange[0] + 0.5*dx;
  for (size_t index = 0; index < numberOfIntervals; ++index) {
    const double middleOfInterval = base + index * dx;
    integral += std::sin(middleOfInterval);
  }

  return integral;
}

double
calculateIntegral_stdThread(const unsigned int numberOfThreads,
                            const unsigned int numberOfIntervals,
                            const std::array<double, 2> & integrationRange) {

        const double dx = (integrationRange[1] - integrationRange[0]) / numberOfIntervals;
        const unsigned intervals_per_thread = numberOfIntervals/numberOfThreads;
        const unsigned intervals_last_thread = numberOfIntervals - intervals_per_thread*(numberOfThreads - 1);
        const double range_length_per_thread = dx*intervals_per_thread;
        const double range_length_last_thread = dx*intervals_last_thread;

        std::vector<double> integrals(numberOfThreads, 0);
        std::vector<std::thread> threads(numberOfThreads);

        for (auto i = 0u; i < numberOfThreads; ++i) {
                const double start = integrationRange[0] + range_length_per_thread*i;
                std::array<double, 2> range = {start, start + (i == numberOfThreads - 1
                                                               ? range_length_last_thread
                                                               : range_length_per_thread)};

                const unsigned nr_intervals = i == numberOfThreads - 1
                        ? intervals_last_thread : intervals_per_thread;

                threads.at(i) = std::thread([=, &integrals](){
                                integrals.at(i) = calculateIntegral_serial(0, nr_intervals, range);
                        });
        }

        for (auto& t : threads)
                t.join();

        return std::accumulate(integrals.begin(), integrals.end(), double(0),
                               std::plus<double>());
}

double
calculateIntegral_omp(const unsigned int numberOfThreads,
                      const unsigned int numberOfIntervals,
                      const std::array<double, 2> & integrationRange) {

        omp_set_num_threads(numberOfThreads);

        double integral = 0;
        const double dx = (integrationRange[1] - integrationRange[0]) / numberOfIntervals;
        const double base = integrationRange[0] + 0.5*dx;

#pragma omp parallel for reduction(+:integral)
        for (size_t index = 0; index < numberOfIntervals; ++index) {
                const double middleOfInterval = base + index * dx;
                integral += std::sin(middleOfInterval);
        }

        return integral;
}

#endif // MAIN2_FUNCTIONS_H
