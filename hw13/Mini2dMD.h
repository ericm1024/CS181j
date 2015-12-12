// -*- C++ -*-
// Mini2dMD.h
// cs181j hw13
// header file containing utility functions

#ifndef MINI2DMD_H
#define MINI2DMD_H

// c junk
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <unistd.h>

// c++ junk
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <string>
#include <array>
#include <fstream>

// include sean's CellArrayNeighbors
#include <geom/orq/CellArrayNeighbors.h>

#include "../Utilities.h"

char exceptionBuffer[10000];
#define throwException(s, ...)                                  \
  sprintf(exceptionBuffer, "%s:%s:%d: " s, __FILE__, __func__,  \
          __LINE__, ##__VA_ARGS__);                             \
  throw std::runtime_error(exceptionBuffer);

using std::string;
using std::vector;
using std::array;

typedef array<double, 2> Point;
typedef array<double, 2> Vector;

struct PointPositionExtractor :
  public std::unary_function<vector<Point>::const_iterator, Point > {
  result_type
  operator()(argument_type r) {
    return *r;
  }
};

namespace Mini2dMD {

struct BoundingBox {
  Point _lower;
  Point _upper;

  BoundingBox(const Point lower, const Point upper) :
    _lower(lower), _upper(upper) {
  }

  bool
  pointIsIn(const Point & p) const {
    return
      (p[0] >= _lower[0]) && (p[0] < _upper[0]) &&
      (p[1] >= _lower[1]) && (p[1] < _upper[1]);
  }

  // convention:
  // this function returns an array of 4 bools which
  //  indicate whether the point is near the boundaries of the box.
  // the convention is to return [0, 1, 2, 3] where 0, 1, 2, and 3
  //  correspond to the following boundaries
  //         3
  //    ------------
  //    |          |
  //  0 | this box | 1
  //    |          |
  //    ------------
  //         2

  // for example, if the return value is [0, 1, 0, 1] then
  //  point is within the specified tolerance of both the right
  //  and the top boundaries.
  // it is not possible to return [0, 0, 1, 1] unless the specified
  //  tolerance is more than half the height of the box.
  array<bool, 4>
  computeBoundaryProximity(const double tolerance,
                           const Point point) const {
    array<bool, 4> proximities;
    proximities[0] = std::abs(_lower[0] - point[0]) < tolerance;
    proximities[1] = std::abs(_upper[0] - point[0]) < tolerance;
    proximities[2] = std::abs(_lower[1] - point[1]) < tolerance;
    proximities[3] = std::abs(_upper[1] - point[1]) < tolerance;
    return proximities;
  }
};

// this function finds the neighbors of all of the positions in
//  the input array that are within a given cutoff radius.
// the neighbors are stored in neighborhoods, one vector of point indices
//  per input position.
// you shouldn't need to modify this.
void
findNeighbors(const vector<Point> & positions,
              const double cutoffRadius,
              vector<vector<unsigned int> > * neighborhoods) {

  geom::CellArrayNeighbors<double, 2, vector<Point>::const_iterator,
                           PointPositionExtractor> neighborSearch;

  neighborSearch.initialize(positions.begin(), positions.end());

  neighborhoods->resize(positions.size());
  vector<vector<Point>::const_iterator > neighborsWithinBall;
  for (unsigned int pointIndex = 0; pointIndex < positions.size();
       ++pointIndex) {
    neighborhoods->at(pointIndex).resize(0);
    neighborsWithinBall.resize(0);
    neighborSearch.neighborQuery(positions[pointIndex], cutoffRadius,
                                 &neighborsWithinBall);
    // For each neighbor of this point.
    for (unsigned int neighborNumber = 0;
         neighborNumber < neighborsWithinBall.size(); ++neighborNumber) {
      const unsigned int neighborIndex =
        std::distance(positions.begin(), neighborsWithinBall[neighborNumber]);
      if (neighborIndex != pointIndex) {
        neighborhoods->at(pointIndex).push_back(neighborIndex);
      }
    }
  }
}

// this function generates the initial conditions of the simulation.
// you shouldn't need to modify this.
template <class RandomNumberEngine>
void
generateInitialCondition(const BoundingBox & simulationDomain,
                         const unsigned int globalNumberOfPoints,
                         const double initialVelocityMaxMagnitude,
                         RandomNumberEngine * randomNumberEngine,
                         vector<Point> * positions,
                         vector<Vector> * velocities) {
  std::uniform_real_distribution<double> uniformRealGenerator(0, 1);
  const double meanVelocityAngle =
    uniformRealGenerator(*randomNumberEngine) * 2 * M_PI;
  const double domainWidth =
    simulationDomain._upper[0] - simulationDomain._lower[0];
  const double domainHeight =
    simulationDomain._upper[1] - simulationDomain._lower[1];

  const unsigned int numberOfPointsPerRow =
    std::ceil(std::sqrt(domainWidth / double(domainHeight) * globalNumberOfPoints));
  const unsigned int numberOfPointsPerCol =
    std::ceil(domainHeight / domainWidth * numberOfPointsPerRow);
  std::normal_distribution<double> normalGenerator(meanVelocityAngle, M_PI/2);
  for (unsigned int globalPointIndex = 0;
       globalPointIndex < globalNumberOfPoints; ++globalPointIndex) {
    const double yOffset =
      (0.5 + (globalPointIndex / numberOfPointsPerRow)) *
      domainHeight / numberOfPointsPerCol;
    const double xOffset =
      (0.5 + (globalPointIndex % numberOfPointsPerRow)) *
      domainWidth / numberOfPointsPerRow;
    const Point initialPoint =
      {{simulationDomain._lower[0] + xOffset,
        simulationDomain._lower[1] + yOffset}};
    const double angle = normalGenerator(*randomNumberEngine);
    const double velocityMagnitude =
      uniformRealGenerator(*randomNumberEngine) * initialVelocityMaxMagnitude;
    const Vector initialVelocity =
      {{velocityMagnitude * cos(angle), velocityMagnitude * sin(angle)}};
    positions->push_back(initialPoint);
    velocities->push_back(initialVelocity);
  }
}

// this is a utility function to make sure that all positions, velocities, and
//  accelerations are finite.
void
checkForInvalidValuesInState(const vector<Point> & positions,
                             const vector<Vector> & velocities,
                             const vector<Vector> & accelerations) {
  for (unsigned int pointIndex = 0;
       pointIndex < positions.size(); ++pointIndex) {
    if (std::isfinite(velocities[pointIndex][0]) == false ||
        std::isfinite(velocities[pointIndex][1]) == false ||
        std::isfinite(positions[pointIndex][0]) == false ||
        std::isfinite(positions[pointIndex][1]) == false ||
        std::isfinite(accelerations[pointIndex][0]) == false ||
        std::isfinite(accelerations[pointIndex][1]) == false) {
      throwException("invalid position = (%11.4e, %11.4e) "
                     "velocity = (%11.4e, %11.4e) or "
                     "acceleration = (%11.4e, %11.4e) for point %u\n",
                     positions[pointIndex][0], positions[pointIndex][1],
                     velocities[pointIndex][0], velocities[pointIndex][1],
                     accelerations[pointIndex][0], accelerations[pointIndex][1],
                     pointIndex);
    }
    // this is a horrible, nasty magic number
    const double magicBadAmountOfMovement = 100;
    if (magnitude(velocities[pointIndex]) > magicBadAmountOfMovement) {
      throwException("velocity = (%11.4e, %11.4e) (magnitude %8.2e) "
                     "just seems too fast for "
                     "point %u, things are probably exploding\n",
                     velocities[pointIndex][0], velocities[pointIndex][1],
                     magnitude(velocities[pointIndex]), pointIndex);
    }
  }
}

// this function is kind of scary.
// it calculates forces and new velocities.
// the input positions and velocities must have the same size and can include
//  shadow particles.  neighborhoods may have indices up to the positions.size().
// a bunch of constants are used for the bogus material model.
// newVelocities and accelerations must be the same size and don't have to
//  be the size of positions and velocities.  they are only for owned particles.
void
calculateForcesAndNewVelocities(const vector<Point> & positions,
                                const vector<Vector> & velocities,
                                const vector<vector<unsigned int> > & neighborhoods,
                                const double cutoffRadius,
                                const double equilibriumDistance,
                                const double forceConstant,
                                const double stickingTimescale,
                                const double mass,
                                const double timestep,
                                const unsigned int startingIndex,
                                const unsigned int endingIndex,
                                vector<Vector> * newVelocities,
                                vector<Vector> * accelerations) {
  vector<Vector> tempVelocities = *newVelocities;
  // calculate forces on our positions
  for (unsigned int pointIndex = startingIndex;
       pointIndex < endingIndex; ++pointIndex) {
    const Point thisPoint = positions[pointIndex];
    Vector thisPointsForce = {{0, 0}};
    Vector averageVelocityOfNeighbors = velocities[pointIndex];
    unsigned int numberOfVelocitiesToAverage = 1;
    // for every neighbor
    for (const unsigned int neighborIndex : neighborhoods[pointIndex]) {
      // calculate the distance
      const Vector diff = thisPoint - positions[neighborIndex];
      const double r = magnitude(diff);
      if (r < 1e-10) {
        throwException("invalid separation "
                       "radius = %8.2e for point %u with neighbor index %u, "
                       "position is (%11.4e, %11.4e)\n",
                       r, pointIndex, neighborIndex,
                       thisPoint[0], thisPoint[1]);
      }
      // if the point is close enough
      if (r < cutoffRadius) {
        // calculate the potential
        const double sigmaOverRSixth =
          pow(equilibriumDistance / r, 6);
        const double forceScalar =
          -1. * forceConstant * sigmaOverRSixth / r * (6. - 12. * sigmaOverRSixth);
        const Vector unitVector = diff / r;
        // add the force
        thisPointsForce += forceScalar * unitVector;
        if (r < equilibriumDistance) {
          averageVelocityOfNeighbors += velocities[neighborIndex];
          ++numberOfVelocitiesToAverage;
        }
      }
    }
    if (neighborhoods[pointIndex].size() > 0) {
      averageVelocityOfNeighbors /= numberOfVelocitiesToAverage;
    }

    // calculate the new velocity
    const Vector velocityWithoutSticking =
      velocities[pointIndex] +
      0.5 * (accelerations->at(pointIndex) + thisPointsForce / mass) * timestep;
    const Vector newVelocity =
      (velocityWithoutSticking +
       timestep / stickingTimescale * averageVelocityOfNeighbors) /
      (1 + timestep / stickingTimescale);
    if (std::isfinite(newVelocity[0]) == false ||
        std::isfinite(newVelocity[1]) == false ||
        magnitude(newVelocity) > 1e4) {
      throwException("when calculating forces we got "
                     "an invalid velocity = (%11.4e, %11.4e) or "
                     "thisPointsForce = (%11.4e, %11.4e) for point %u at "
                     "(%11.4e, %11.4e) with velocityWithoutSticking = "
                     "(%11.4e, %11.4e) and accelerations[%u] = (%11.4e, %11.4e) and "
                     "averageVelocityOfNeighbors = (%11.4e, %11.4e)\n",
                     newVelocity[0], newVelocity[1],
                     thisPointsForce[0], thisPointsForce[1],
                     pointIndex,
                     thisPoint[0], thisPoint[1],
                     velocityWithoutSticking[0], velocityWithoutSticking[1],
                     pointIndex,
                     accelerations->at(pointIndex)[0], accelerations->at(pointIndex)[1],
                     averageVelocityOfNeighbors[0], averageVelocityOfNeighbors[1]);
    }
    tempVelocities[pointIndex] = newVelocity;

    // assign accelerations for the next timestep
    accelerations->at(pointIndex) = thisPointsForce / mass;

  }
  *newVelocities = tempVelocities;
}

// this function updates positions given their velocities and accelerations,
//  then applies the wall reflection condition.
void
integrateVelocitiesAndReflectParticlesOnWalls(const BoundingBox & simulationDomain,
                                              const vector<Vector> & accelerations,
                                              const unsigned int startingIndex,
                                              const unsigned int endingIndex,
                                              const double timestep,
                                              vector<Vector> * velocities,
                                              vector<Point> * positions) {

  // make sure that nothing starts as nan
  try {
    Mini2dMD::checkForInvalidValuesInState(*positions,
                                           *velocities,
                                           accelerations);
  } catch (const std::exception & e) {
    fprintf(stderr, "invalid value in state at start of "
            "integrateVelocitiesAndReflectParticlesOnWalls\n");
    throw;
  }

  for (unsigned int pointIndex = startingIndex;
       pointIndex < endingIndex; ++pointIndex) {
    // integrate velocities to get positions
    positions->at(pointIndex) +=
      velocities->at(pointIndex) * timestep +
      0.5 * accelerations[pointIndex] * timestep * timestep;

    // if points are outside of the domain,
    //  reflect the velocities on the boundaries and bring the points back
    //  to the domain wall
    for (unsigned int coordinate = 0; coordinate < 2; ++coordinate) {
      if (positions->at(pointIndex)[coordinate] >=
          simulationDomain._upper[coordinate]) {
        positions->at(pointIndex)[coordinate] =
          simulationDomain._upper[coordinate] - 1e-6;
        if (velocities->at(pointIndex)[coordinate] > 0) {
          velocities->at(pointIndex)[coordinate] =
            -1 * std::abs(velocities->at(pointIndex)[coordinate]);
        }
      }
      if (positions->at(pointIndex)[coordinate] <=
          simulationDomain._lower[coordinate]) {
        positions->at(pointIndex)[coordinate] =
          simulationDomain._lower[coordinate] + 1e-6;
        if (velocities->at(pointIndex)[coordinate] < 0) {
          velocities->at(pointIndex)[coordinate] =
            std::abs(velocities->at(pointIndex)[coordinate]);
        }
      }
    }
  }

  // make sure that nothing has become nan
  try {
    Mini2dMD::checkForInvalidValuesInState(*positions,
                                           *velocities,
                                           accelerations);
  } catch (const std::exception & e) {
    fprintf(stderr, "invalid value in state at start of "
            "integrateVelocitiesAndReflectParticlesOnWalls\n");
    throw;
  }

}

}

#endif // MINI2DMD_H
