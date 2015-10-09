// -*- C++ -*-
// Main0.cc
// cs181j 2015 hw5 example
// This program shows how to use vectorization to accelerate finding
//  the index of an extremum in an array.

// special include file for SIMD commands
#include <immintrin.h>

#include <random>
#include <array>
#include <chrono>
#include <vector>

using std::vector;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

unsigned int
findIndexOfMinimumValue_scalar(const float * input,
                               const unsigned int inputSize) {

  unsigned int indexOfMinimumValue = 0;
  float minimumValue = input[0];

  for (unsigned int i = 0; i < inputSize; ++i) {
    asm("# keep your grubby mitts out of this loop, autovectorizer!");
    const float value = input[i];
    if (value < minimumValue) {
      indexOfMinimumValue = i;
      minimumValue = value;
    }
  }

  return indexOfMinimumValue;
}

unsigned int
findIndexOfMinimumValue_manual(const float * input,
                               const unsigned int inputSize) {

  // Process the majority of the input with vector registers.  Each of
  //  the vector lanes will keep track of its own minimum, then at the
  //  end we'll take the minimum of all the lanes.
  // This means that we have a register for the indices of the minima...
  __m128i simdIndexOfMinimumValue = _mm_set1_epi32(0);
  //  ...and a register for the minima themselves.
  __m128 simdMinimumValue = _mm_set1_ps(input[0]);
  // This register stores the input indices for the the current loop
  //  iteration.  I add to it on each iteration rather than using set.
  __m128i simdInputIndices = _mm_set_epi32(3, 2, 1, 0);
  // This is the increment for the simd indices, which just increases
  //  the indices by four each time.
  const __m128i simdIndicesIncrement = _mm_set1_epi32(4);
  // For each group of four
  unsigned int inputIndex = 0;
  for (; inputIndex < (inputSize & ~3); inputIndex += 4) {
    // Pull the values out of the input
    const __m128 simdValues = _mm_load_ps(&input[inputIndex]);
    // Compare the values with the current minima.  This produces
    //  a mask of ones where the condition is true and zeroes otherwise.
    // That is, if ints were 4 bits and we had 16-bit simd registers
    //  and we compared the two registers
    //  [ 7, 8, 9, 10]
    //  and
    //  [ 8, 2, 9, 11]
    //  the mask would be
    //  [0000 1111 0000 0000]
    const __m128 simdMask = _mm_cmplt_ps(simdValues, simdMinimumValue);
    // Now, this line is a bit of magic, using the blend operation.
    // Blend takes some values from one register and some from another,
    //  depending on the mask.  If the mask is 1 somewhere (which
    //  means that the new values are lower than the current mimima),
    //  then that lane's index of the minimum is replaced by the new index.
    simdIndexOfMinimumValue =
      _mm_blendv_epi8(simdIndexOfMinimumValue, simdInputIndices,
                      __m128i(simdMask));
    // Increment the indices
    simdInputIndices += simdIndicesIncrement;
    // Do the same blending, but on the actual values and not just the indices.
    simdMinimumValue =
      _mm_blendv_ps(simdMinimumValue, simdValues, simdMask);
  }
  // Get the minimum out of the simd vector
  int * intTemp = (int*)&simdIndexOfMinimumValue[0];
  float minimumValue = simdMinimumValue[0];
  unsigned int indexOfMinimumValue = intTemp[0];
  for (unsigned int i = 1; i < 4; ++i) {
    const float value = simdMinimumValue[i];
    if (value < minimumValue) {
      indexOfMinimumValue = intTemp[i];
      minimumValue = value;
    }
  }

  // Now process the remainder that didn't divide the vector size evenly
  for (; inputIndex < inputSize; ++inputIndex) {
    const float value = input[inputIndex];
    if (value < minimumValue) {
      indexOfMinimumValue = inputIndex;
      minimumValue = value;
    }
  }
  return indexOfMinimumValue;

}

template <class Function>
void
runTimingTest(const unsigned int numberOfTrials,
              const Function function,
              const float * input,
              const unsigned int inputSize,
              unsigned int * indexOfMinimumValue,
              float * elapsedTime) {

  *elapsedTime = std::numeric_limits<float>::max();

  for (unsigned int trialNumber = 0;
       trialNumber < numberOfTrials; ++trialNumber) {

    // Start measuring
    const high_resolution_clock::time_point tic = high_resolution_clock::now();

    // Do the clustering
    *indexOfMinimumValue = function(input, inputSize);

    // Stop measuring
    const high_resolution_clock::time_point toc = high_resolution_clock::now();
    const float thisTrialsElapsedTime =
      duration_cast<duration<float> >(toc - tic).count();
    // Take the minimum values from all trials
    *elapsedTime = std::min(*elapsedTime, thisTrialsElapsedTime);
  }

}

int main() {

  // ===========================================================================
  // *************************** < Inputs> *************************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  const unsigned int inputSize = 1e8;
  const unsigned int numberOfTrials = 10;

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // *************************** </Inputs> *************************************
  // ===========================================================================

  // Make a random number generator
  std::default_random_engine randomNumberEngine;
  std::uniform_real_distribution<float> randomNumberGenerator(0, 1);

  // Generate input
  vector<float> input(inputSize);
  for (unsigned int i = 0; i < inputSize; ++i) {
    input[i] = randomNumberGenerator(randomNumberEngine);
  }

  // Run the scalar version
  float elapsedTime_scalar;
  unsigned int indexOfMinimumValue_scalar;
  runTimingTest(numberOfTrials,
                findIndexOfMinimumValue_scalar,
                &input[0],
                inputSize,
                &indexOfMinimumValue_scalar,
                &elapsedTime_scalar);

  // Run the vectorized version
  float elapsedTime_manual;
  unsigned int indexOfMinimumValue_manual;
  runTimingTest(numberOfTrials,
                findIndexOfMinimumValue_manual,
                &input[0],
                inputSize,
                &indexOfMinimumValue_manual,
                &elapsedTime_manual);

  if (indexOfMinimumValue_scalar != indexOfMinimumValue_manual) {
    fprintf(stderr, "inconsistent answers, \n"
            "scalar gets %9u (%12.6e), \n"
            "manual gets %9u (%12.6e), \n",
            indexOfMinimumValue_scalar,
            input[indexOfMinimumValue_scalar],
            indexOfMinimumValue_manual,
            input[indexOfMinimumValue_manual]);
    exit(1);
  }
  printf("speedup from vectorization is %5.2lf\n",
         elapsedTime_scalar / elapsedTime_manual);

  return 0;
}
