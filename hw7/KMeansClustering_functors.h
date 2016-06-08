// -*- C++ -*-
#ifndef KMEANS_FUNCTORS_H
#define KMEANS_FUNCTORS_H

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

#include <vector>
#include <iostream>
#include <omp.h>
#include <atomic>
#include <cassert>
#include <type_traits>

#include "aligned_allocator.hpp"

struct kmc_base {
        std::string name_;

        kmc_base(const char* name) : name_{name} {}
        
        virtual void calculateClusterCentroids(const unsigned, const unsigned,
                                               const std::vector<Point>&,
                                               const std::vector<Point>&,
                                               std::vector<Point>*) {}
        virtual ~kmc_base() = default;
};

struct
SerialClusterer : kmc_base {

        std::vector<Point> _centroids;
        std::vector<Point> _nextCentroids;
        std::vector<unsigned int> _nextCentroidCounts;

        SerialClusterer(const unsigned int numberOfCentroids) :
                kmc_base{"serial"},
                _centroids(numberOfCentroids),
                _nextCentroids(numberOfCentroids),
                _nextCentroidCounts(numberOfCentroids) {
        }

        void
        calculateClusterCentroids(const unsigned int ignoredNumberOfThreads,
                                  const unsigned int numberOfIterations,
                                  const std::vector<Point> & points,
                                  const std::vector<Point> & startingCentroids,
                                  std::vector<Point> * finalCentroids) override {
                
                ignoreUnusedVariable(ignoredNumberOfThreads);

                const unsigned int numberOfPoints = points.size();
                const unsigned int numberOfCentroids = startingCentroids.size();

                _centroids = startingCentroids;

                // Start with values of 0 for the next centroids
                std::fill(_nextCentroids.begin(), _nextCentroids.end(),
                          (Point) {{0., 0., 0.}});
                std::fill(_nextCentroidCounts.begin(), _nextCentroidCounts.end(), 0);

                // For each of a fixed number of iterations
                for (unsigned int iterationNumber = 0;
                     iterationNumber < numberOfIterations; ++iterationNumber) {
                        // Calculate next centroids
                        for (unsigned int pointIndex = 0;
                             pointIndex < numberOfPoints; ++pointIndex) {
                                const Point & point = points[pointIndex];
                                // Find which centroid this point is closest to
                                unsigned int indexOfClosestCentroid = 0;
                                // First centroid is considered the closest to start
                                double squaredDistanceToClosestCentroid =
                                        squaredMagnitude(point - _centroids[0]);
                                // For each centroid after the first
                                for (unsigned int centroidIndex = 1;
                                     centroidIndex < numberOfCentroids; ++centroidIndex) {
                                        const double squaredDistanceToThisCentroid =
                                                squaredMagnitude(point - _centroids[centroidIndex]);
                                        // If we're closer, change the closest one
                                        if (squaredDistanceToThisCentroid < squaredDistanceToClosestCentroid) {
                                                indexOfClosestCentroid = centroidIndex;
                                                squaredDistanceToClosestCentroid = squaredDistanceToThisCentroid;
                                        }
                                }

                                // Add our point to the next centroid value
                                _nextCentroids[indexOfClosestCentroid] += point;
                                ++_nextCentroidCounts[indexOfClosestCentroid];
                        }

                        // Move centroids
                        for (unsigned int centroidIndex = 0;
                             centroidIndex < numberOfCentroids; ++centroidIndex) {
                                // The next centroid value is the average of the points that were
                                //  closest to it.
                                _centroids[centroidIndex] =
                                        _nextCentroids[centroidIndex] / _nextCentroidCounts[centroidIndex];
                                // Reset the intermediate values
                                _nextCentroidCounts[centroidIndex] = 0;
                                _nextCentroids[centroidIndex] = (Point) {{0., 0., 0.}};
                        }
                }
                *finalCentroids = _centroids;
        }

};

struct ThreadedClusterer : kmc_base {
        
        std::vector<Point> _centroids;
        std::vector<Point> _nextCentroids;
        std::vector<unsigned> _nextCentroidCounts;

        ThreadedClusterer(const unsigned int numberOfCentroids) :
                kmc_base{"omp_locked"},
                _centroids(numberOfCentroids),
                _nextCentroids(numberOfCentroids),
                _nextCentroidCounts(numberOfCentroids) {
        }

        void
        calculateClusterCentroids(const unsigned int numberOfThreads,
                                  const unsigned int numberOfIterations,
                                  const std::vector<Point> & points,
                                  const std::vector<Point> & startingCentroids,
                                  std::vector<Point> * finalCentroids) override {
                
                omp_set_num_threads(numberOfThreads);
                
                const unsigned int numberOfPoints = points.size();
                const unsigned int numberOfCentroids = startingCentroids.size();
          
                _centroids = startingCentroids;
                
                // Start with values of 0 for the next centroids
                std::fill(_nextCentroids.begin(), _nextCentroids.end(), (Point) {{0., 0., 0.}});
                std::fill(_nextCentroidCounts.begin(), _nextCentroidCounts.end(), 0);

                for (auto n = 0u; n < numberOfIterations; ++n) {
                        // Calculate next centroids
                        #pragma omp parallel for
                        for (auto i = 0u; i < numberOfPoints; ++i) {
                                const auto& point = points[i];

                                unsigned int closest_idx = 0;
                                double closest_dist = squaredMagnitude(point - _centroids[0]);

                                // For each centroid after the first
                                for (auto j = 1u; j < numberOfCentroids; ++j) {
                                        const double squaredDistanceToThisCentroid =
                                                squaredMagnitude(point - _centroids[j]);

                                        // If we're closer, change the closest one
                                        if (squaredDistanceToThisCentroid < closest_dist) {
                                                closest_idx = j;
                                                closest_dist = squaredDistanceToThisCentroid;
                                        }
                                }

#pragma omp critical
                                {
                                        // Add our point to the next centroid value
                                        _nextCentroids[closest_idx] += point;
                                        ++_nextCentroidCounts[closest_idx];
                                }
                        }

                        // Move centroids
                        for (auto i = 0u; i < numberOfCentroids; ++i) {
                                // The next centroid value is the average of the points that were
                                //  closest to it.
                                _centroids[i] = _nextCentroids[i] / _nextCentroidCounts[i];
                                // Reset the intermediate values
                                _nextCentroidCounts[i] = 0;
                                _nextCentroids[i] = (Point) {{0., 0., 0.}};
                        }
                }
                *finalCentroids = _centroids;
        }
};

template <typename T, std::size_t allign>
struct alignas(allign) _alligned : public T {
private:
        std::array<std::uint8_t, sizeof(T)%allign == 0 ? 0 : allign - sizeof(T)%allign> __pad;

public:
        template <typename ... Ts>
        _alligned(Ts&& ... args) : T{std::forward<Ts>(args)...}, __pad{{0}}
        {}

        template <typename ... Ts>
        _alligned& operator=(Ts&& ... args)
        {
                T::operator=(std::forward<Ts>(args)...);
                return *this;
        }
};

// this should really use C++11 atomics, but that would require re-writing Point, so...
static inline void atomic_add_point(Point& sum, const Point& addend)
{
        for (auto i = 0u; i < sum.size(); ++i) {
                double old_sum, new_sum;
                do {
                        old_sum = sum[i];
                        new_sum = old_sum + addend[i];
                } while (!(((std::atomic<double>*)(&sum[i]))-> // RIP
                           compare_exchange_weak(old_sum, new_sum)));
        }
}

template <bool should_allign>
struct ThreadedClusterer_atomic_base : kmc_base {

        static constexpr std::size_t cache_line_size{64};
        
        std::vector<Point> _centroids;

        template <typename T>
        using alloc_t = aligned_allocator<T, cache_line_size>;

        using next_centroid_value_type = typename std::conditional<should_allign,
                                                                   _alligned<Point, cache_line_size>,
                                                                   Point>::type;
        std::vector<next_centroid_value_type, alloc_t<next_centroid_value_type>> _nextCentroids;

        using counts_value_type = typename std::conditional<should_allign,
                                                            _alligned<std::atomic<unsigned>, cache_line_size>,
                                                            std::atomic<unsigned>>::type;
        std::vector<counts_value_type, alloc_t<counts_value_type>> _nextCentroidCounts;

        ThreadedClusterer_atomic_base(const unsigned int numberOfCentroids) :
                kmc_base{should_allign ? "omp_atomic" : "omp_atomic_false_sharing"},
                _centroids(numberOfCentroids),
                _nextCentroids(numberOfCentroids),
                _nextCentroidCounts(numberOfCentroids)
        {}

        void
        calculateClusterCentroids(const unsigned int numberOfThreads,
                                  const unsigned int numberOfIterations,
                                  const std::vector<Point> & points,
                                  const std::vector<Point> & startingCentroids,
                                  std::vector<Point> * finalCentroids) override {
                
                omp_set_num_threads(numberOfThreads);
                
                const unsigned int numberOfPoints = points.size();
                const unsigned int numberOfCentroids = startingCentroids.size();
          
                _centroids = startingCentroids;
                
                // Start with values of 0 for the next centroids
                std::fill(_nextCentroids.begin(), _nextCentroids.end(), (Point) {{0., 0., 0.}});
                std::fill(_nextCentroidCounts.begin(), _nextCentroidCounts.end(), 0);

                for (auto n = 0u; n < numberOfIterations; ++n) {
                        // Calculate next centroids
                        #pragma omp parallel for 
                        for (auto i = 0u; i < numberOfPoints; ++i) {
                                const auto& point = points[i];

                                unsigned int closest_idx = 0;
                                double closest_dist = squaredMagnitude(point - _centroids[0]);

                                // For each centroid after the first
                                for (auto j = 1u; j < numberOfCentroids; ++j) {
                                        const double squaredDistanceToThisCentroid =
                                                squaredMagnitude(point - _centroids[j]);

                                        // If we're closer, change the closest one
                                        if (squaredDistanceToThisCentroid < closest_dist) {
                                                closest_idx = j;
                                                closest_dist = squaredDistanceToThisCentroid;
                                        }
                                }

                                // Add our point to the next centroid value
                                atomic_add_point(_nextCentroids[closest_idx], point);
                                ++_nextCentroidCounts[closest_idx];
                        }

                        // Move centroids
                        for (auto i = 0u; i < numberOfCentroids; ++i) {
                                // The next centroid value is the average of the points that were
                                //  closest to it.
                                _centroids[i] = _nextCentroids[i] / _nextCentroidCounts[i];
                                // Reset the intermediate values
                                _nextCentroidCounts[i] = 0.;
                                _nextCentroids[i] = (Point) {{0., 0., 0.}};
                        }
                }
                *finalCentroids = _centroids;
        }
};


using ThreadedClusterer2 = ThreadedClusterer_atomic_base<true>; // allgined
using ThreadedClusterer5 = ThreadedClusterer_atomic_base<false>; // not alligned

struct ThreadedClusterer3 : kmc_base {

        std::vector<Point> _centroids;
        std::vector<Point> _nextCentroids;
        std::vector<unsigned> _nextCentroidCounts;

        ThreadedClusterer3(const unsigned int numberOfCentroids) :
                kmc_base{"omp_thread_local"},
                _centroids(numberOfCentroids),
                _nextCentroids(numberOfCentroids),
                _nextCentroidCounts(numberOfCentroids) {
        }

        void
        calculateClusterCentroids(const unsigned int numberOfThreads,
                                  const unsigned int numberOfIterations,
                                  const std::vector<Point> & points,
                                  const std::vector<Point> & startingCentroids,
                                  std::vector<Point> * finalCentroids) override {
                
                omp_set_num_threads(numberOfThreads);
                
                const unsigned int numberOfPoints = points.size();
                const unsigned int numberOfCentroids = startingCentroids.size();
          
                _centroids = startingCentroids;
                
                // Start with values of 0 for the next centroids
                std::fill(_nextCentroids.begin(), _nextCentroids.end(), (Point) {{0., 0., 0.}});
                std::fill(_nextCentroidCounts.begin(), _nextCentroidCounts.end(), 0);

                for (auto n = 0u; n < numberOfIterations; ++n) {
                        #pragma omp parallel
                        {
                                std::vector<Point> local_next_centroids(_nextCentroids.size(), {{0, 0, 0}});
                                std::vector<unsigned> local_next_counts(_nextCentroidCounts.size(), 0);

                                // Calculate next centroids
                                #pragma omp for
                                for (auto i = 0u; i < numberOfPoints; ++i) {
                                        const auto& point = points[i];

                                        unsigned int closest_idx = 0;
                                        double closest_dist = squaredMagnitude(point - _centroids[0]);

                                        // For each centroid after the first
                                        for (auto j = 1u; j < numberOfCentroids; ++j) {
                                                const double squaredDistanceToThisCentroid =
                                                        squaredMagnitude(point - _centroids[j]);

                                                // If we're closer, change the closest one
                                                if (squaredDistanceToThisCentroid < closest_dist) {
                                                        closest_idx = j;
                                                        closest_dist = squaredDistanceToThisCentroid;
                                                }
                                        }

                                        // Add our point to the next centroid value
                                        local_next_centroids[closest_idx] += point;
                                        ++local_next_counts[closest_idx];
                                }

                                // add this thread's contribution to _nextCentroids and _nextCentroidCounts
                                #pragma omp critical
                                {
                                        for (auto i = 0u; i < _nextCentroids.size(); ++i) {
                                                _nextCentroids[i] += local_next_centroids[i];
                                                _nextCentroidCounts[i] += local_next_counts[i];
                                        }
                                }
                        }

                        // Move centroids
                        for (auto i = 0u; i < numberOfCentroids; ++i) {
                                // The next centroid value is the average of the points that were
                                //  closest to it.
                                _centroids[i] = _nextCentroids[i] / _nextCentroidCounts[i];
                                // Reset the intermediate values
                                _nextCentroidCounts[i] = 0;
                                _nextCentroids[i] = (Point) {{0., 0., 0.}};
                        }
                }
                *finalCentroids = _centroids;
        }
};

struct ThreadedClusterer4 : kmc_base {

        std::vector<Point> _centroids;
        std::vector<Point> _nextCentroids;
        std::vector<unsigned> _nextCentroidCounts;

        ThreadedClusterer4(const unsigned int numberOfCentroids) :
                kmc_base{"omp_moved_pragma"},
                _centroids(numberOfCentroids),
                _nextCentroids(numberOfCentroids),
                _nextCentroidCounts(numberOfCentroids) {
        }

        void
        calculateClusterCentroids(const unsigned int numberOfThreads,
                                  const unsigned int numberOfIterations,
                                  const std::vector<Point> & points,
                                  const std::vector<Point> & startingCentroids,
                                  std::vector<Point> * finalCentroids) override {
                
                omp_set_num_threads(numberOfThreads);
                
                const unsigned int numberOfPoints = points.size();
                const unsigned int numberOfCentroids = startingCentroids.size();
          
                _centroids = startingCentroids;
                
                // Start with values of 0 for the next centroids
                std::fill(_nextCentroids.begin(), _nextCentroids.end(), (Point) {{0., 0., 0.}});
                std::fill(_nextCentroidCounts.begin(), _nextCentroidCounts.end(), 0);

#pragma omp parallel
                for (auto n = 0u; n < numberOfIterations; ++n) {
                        std::vector<Point> local_next_centroids(_nextCentroids.size(), {{0, 0, 0}});
                        std::vector<unsigned> local_next_counts(_nextCentroidCounts.size(), 0);

                        // Calculate next centroids
#pragma omp for
                        for (auto i = 0u; i < numberOfPoints; ++i) {
                                const auto& point = points[i];
                                
                                unsigned int closest_idx = 0;
                                double closest_dist = squaredMagnitude(point - _centroids[0]);
                                
                                // For each centroid after the first
                                for (auto j = 1u; j < numberOfCentroids; ++j) {
                                        const double squaredDistanceToThisCentroid =
                                                squaredMagnitude(point - _centroids[j]);
                                        
                                        // If we're closer, change the closest one
                                        if (squaredDistanceToThisCentroid < closest_dist) {
                                                closest_idx = j;
                                                closest_dist = squaredDistanceToThisCentroid;
                                        }
                                }
                                
                                // Add our point to the next centroid value
                                local_next_centroids[closest_idx] += point;
                                ++local_next_counts[closest_idx];
                        }

                        // add this thread's contribution to _nextCentroids and _nextCentroidCounts
#pragma omp critical
                        {
                                for (auto i = 0u; i < _nextCentroids.size(); ++i) {
                                        _nextCentroids[i] += local_next_centroids[i];
                                        _nextCentroidCounts[i] += local_next_counts[i];
                                }
                        }

#pragma omp barrier
                        
                        // Move centroids
#pragma omp for
                        for (auto i = 0u; i < numberOfCentroids; ++i) {
                                // The next centroid value is the average of the points that were
                                //  closest to it.
                                _centroids[i] = _nextCentroids[i] / _nextCentroidCounts[i];
                                // Reset the intermediate values
                                _nextCentroidCounts[i] = 0;
                                _nextCentroids[i] = (Point) {{0., 0., 0.}};
                        }
                        *finalCentroids = _centroids;
                }
        }
};

#endif // KMEANS_FUNCTORS_H
