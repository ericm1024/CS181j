// -*- C++ -*-
// KMeansClustering.cc
// cs181j 2015 hw7
// An exercise in threading, accelerating the computation of k-means
//  clustering.

// These utilities are used on many assignments
#include "../Utilities.h"

// Many of the homework assignments have definitions that are common across
//  several executables, so we group them together
#include "CommonDefinitions.h"

#include "KMeansClustering_functors.h"

#include <map>
#include <utility>
#include <memory>

using std::make_shared;
using std::make_pair;
using std::map;
using std::vector;
using std::array;
using std::string;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

void
checkResult(const vector<Point> & correctResult,
            const vector<Point> & testResult,
            const string & testName,
            const double absoluteErrorTolerance)
{
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
                const double absoluteError =
                        magnitude(correctResult[i] - testResult[i]);
                if (absoluteError > absoluteErrorTolerance) {
                        sprintf(sprintfBuffer, "wrong result for centroid number %zu in test result, "
                                "it's (%e, %e, %e) but should be (%e, %e, %e), test named "
                                BOLD_ON FG_RED "%s" RESET "\n", i,
                                testResult[i][0], testResult[i][1], testResult[i][2],
                                correctResult[i][0], correctResult[i][1], correctResult[i][2],
                                testName.c_str());
                        throw std::runtime_error(sprintfBuffer);
                }
        }
}

template <class Clusterer>
void
runTimingTest(const unsigned int numberOfTrials,
              const unsigned int numberOfIterations,
              const unsigned int numberOfThreads,
              const vector<Point> & points,
              const vector<Point> & startingCentroids,
              Clusterer * clusterer,
              vector<Point> * finalCentroids,
              double * elapsedTime) {

        
        auto elapsed_local = std::numeric_limits<double>::max();

        for (unsigned int trialNumber = 0;
             trialNumber < numberOfTrials; ++trialNumber) {

                // Reset the final centroids
                (*finalCentroids) = startingCentroids;

                // Start measuring
                const high_resolution_clock::time_point tic = high_resolution_clock::now();

                // Do the clustering
                clusterer->calculateClusterCentroids(numberOfThreads,
                                                     numberOfIterations,
                                                     points,
                                                     startingCentroids,
                                                     finalCentroids);

                // Stop measuring
                const high_resolution_clock::time_point toc = high_resolution_clock::now();
                const double thisTrialsElapsedTime =
                        duration_cast<duration<double> >(toc - tic).count();
                
                // Take the minimum values from all trials
                elapsed_local = std::min(elapsed_local, thisTrialsElapsedTime);
        }

        // some tests just want the results, not the time
        if (elapsedTime)
                *elapsedTime = elapsed_local;
}

int main() {

        // ===========================================================================
        // *************************** < Inputs> *************************************
        // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

        // A lot of homeworks will run something over a range of sizes,
        //  which will then be plotted by some script.
        // This controls how many points are used.
        const array<double, 2> rangeOfNumberOfPoints = {{1e2, 1e6}};
        // This number controls how many data points are made and plotted.
        const unsigned int numberOfPointsDataPoints = 10;
        // This controls how many centroids are used.
        const unsigned int numberOfCentroids = 100;
        const vector<unsigned int> numbersOfThreads    =
                {{1, 2, 4, 6, 8, 10, 11, 12, 13, 14, 16, 20, 22, 24, 26, 28, 30, 36, 42, 48}};
        // In real k-means calculations, the centroid updates would happen
        //  until some condition is satisfied.  In this, we'll just iterate
        //  a fixed number of times, so that all methods do the same amount
        //  of work.
        const unsigned int numberOfIterations = 5;
        // This is the standard number of times the calculation is repeated.
        const unsigned int numberOfTrials = 20;

        std::vector<std::shared_ptr<kmc_base>> test_functors =
                // *** ADD NEW FUNCTORS HERE ***
                {
                        make_shared<SerialClusterer>(numberOfCentroids),
                        //make_shared<ThreadedClusterer>(numberOfCentroids),
                        make_shared<ThreadedClusterer2>(numberOfCentroids),
                        //make_shared<ThreadedClusterer3>(numberOfCentroids),
                        //make_shared<ThreadedClusterer4>(numberOfCentroids),
                        //make_shared<ThreadedClusterer5>(numberOfCentroids),
                };

        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        // *************************** </Inputs> *************************************
        // ===========================================================================

        // On each test, we need to make sure we get the same result.  A test will
        //  fail if the difference between any entry in our result is more than
        //  absoluteErrorTolerance different than entries we got with another method.
        const double absoluteErrorTolerance = 1e-4;

        // Make sure that the data directory exists.
        Utilities::verifyThatDirectoryExists("data");

        // Make a random number generator
        std::default_random_engine randomNumberGenerator;

        // Prepare output matrices
        vector<vector<double> >
                numberOfPointsMatrixForPlotting(numberOfPointsDataPoints,
                                                vector<double>(numbersOfThreads.size(), 0));
        vector<vector<double> >
                numberOfThreadsMatrixForPlotting(numberOfPointsDataPoints,
                                                 vector<double>(numbersOfThreads.size(), 0));

        // lol
        map<string, vector<vector<double>>> times;
        for (const auto& f : test_functors)
                times.insert(make_pair(f->name_,
                                       vector<vector<double>>(numberOfPointsDataPoints,
                                                      vector<double>(numbersOfThreads.size(), 0.))));

        // For each resolution data point
        for (unsigned int nr_points_idx = 0; nr_points_idx < numberOfPointsDataPoints; ++nr_points_idx) {
                // Calculate the number of points so that it's linear on a
                //  log scale.
                const size_t numberOfPoints =
                        Utilities::interpolateNumberLinearlyOnLogScale(rangeOfNumberOfPoints[0],
                                                                       rangeOfNumberOfPoints[1],
                                                                       numberOfPointsDataPoints,
                                                                       nr_points_idx);

                const auto thisSizesTic = high_resolution_clock::now();

                // Prepare real distributions for generating initial points
                const double numberOfCentroidsPerSide = std::ceil(std::pow(numberOfCentroids, 1./3.));
                auto uniform_rand = std::bind(std::uniform_real_distribution<double>(0, 1.),
                                              randomNumberGenerator);
                const double normalMean = (1. / numberOfCentroidsPerSide) / 2.;
                const double normalStandardDeviation = normalMean / 3.;
                auto normal_rand = std::bind(std::normal_distribution<double>(normalMean,
                                                                              normalStandardDeviation),
                                             randomNumberGenerator);
                
                const auto generate_point = [&](auto& gen) -> Point {
                        Point p;
                        std::generate(p.begin(), p.end(), gen);
                        return p;
                };
                
                // Prepare points
                vector<Point> points;
                points.reserve(numberOfPoints);
                const unsigned int numberOfPointsPerCentroid = numberOfPoints / numberOfCentroids;
                for (auto n = 0u; n < numberOfCentroids; ++n) {
                        const Point centroid = generate_point(uniform_rand);
                        for (auto k = 0u; k < numberOfPointsPerCentroid; ++k)
                                points.push_back(centroid + generate_point(normal_rand));
                }

                // Throw in random points until it's full
                while (points.size() != numberOfPoints)
                        points.push_back(generate_point(uniform_rand));

                // Compute starting locations for the centroids
                vector<Point> startingCentroids(numberOfCentroids);
                std::generate(startingCentroids.begin(), startingCentroids.end(),
                              [&](){return generate_point(uniform_rand);});

                vector<Point> serialFinalCentroids(numberOfCentroids);
                double serialElapsedTime;
                runTimingTest(numberOfTrials,
                              numberOfIterations,
                              42, // this is really ignored
                              points,
                              startingCentroids,
                              test_functors[0].get(),
                              &serialFinalCentroids,
                              &serialElapsedTime);

                // for each numberOfThreads
                for (unsigned int nr_threads_idx = 0; nr_threads_idx < numbersOfThreads.size(); ++nr_threads_idx) {

                        // get the number of threads
                        const unsigned int numberOfThreads = numbersOfThreads[nr_threads_idx];

                        // NB: we end up running serial twice but w/e (once up there ^)
                        for (auto& functor : test_functors) {
                                vector<Point> final_centroids(numberOfCentroids);
                                double time;
                                runTimingTest(numberOfTrials,
                                              numberOfIterations,
                                              numberOfThreads,
                                              points,
                                              startingCentroids,
                                              functor.get(),
                                              &final_centroids,
                                              &time);

                                checkResult(serialFinalCentroids,
                                            final_centroids,
                                            functor->name_,
                                            absoluteErrorTolerance);

                                times.at(functor->name_).at(nr_points_idx).at(nr_threads_idx) = time;
                        }

                        numberOfPointsMatrixForPlotting.at(nr_points_idx).at(nr_threads_idx) =
                                numberOfPoints;
                        numberOfThreadsMatrixForPlotting.at(nr_points_idx).at(nr_threads_idx) =
                                numberOfThreads;
                }

                const auto thisSizesToc = high_resolution_clock::now();
                const double thisSizesElapsedTime = duration_cast<duration<double>>(thisSizesToc - thisSizesTic).count();
                printf("processing %8.2e points took %7.2f seconds\n",
                       double(numberOfPoints),
                       thisSizesElapsedTime);
        }

        const string prefix = "data/KMeansClustering_";
        const string suffix = "_shuffler";

        Utilities::writeMatrixToFile(numberOfPointsMatrixForPlotting,
                                     prefix + string("numberOfPoints") + suffix);
        Utilities::writeMatrixToFile(numberOfThreadsMatrixForPlotting,
                                     prefix + string("numberOfThreads") + suffix);

        for (const auto& f : test_functors) 
                Utilities::writeMatrixToFile(times.at(f->name_), prefix + f->name_ + suffix);

        return 0;
}
