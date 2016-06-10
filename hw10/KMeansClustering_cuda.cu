// -*- C++ -*-
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <cfloat>

#include <cuda_runtime.h>

#include "KMeansClustering_cuda.cuh"
#include "../GpuUtilities.h"

template <class T>
__global__
void
setArray_kernel(const unsigned int arraySize,
                const T value,
                T * __restrict__ array) {

  unsigned int arrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
  while (arrayIndex < arraySize) {
    array[arrayIndex] = value;
    arrayIndex += blockDim.x * gridDim.x;
  }
}

// design of GPU k-means clustering implementation:
//     We perform a fixed number of iterations. each iteration is performed
//     using 2 kernels:
//         - The first kernel calculates the next_centroid locations and counts.
//           This calculation is done using block-local shared memory. To reduce
//           the result, the shared memory contents are moved to global memory at
//           the end of the kernel.
//         - The second kernel reduces the block local results and moves the
//           centroids to their location for the next iteration

__global__ static void
find_local_centroids_kernel(const float * points,
                            const unsigned nr_points,
                            float *centroids,
                            const unsigned nr_centroids,
                            float *next_centroids, // size = nr_centroids*3*nr_blocks
                            unsigned *next_centroid_counts) // size = nr_centroids*nr_blocks
{
        extern __shared__ uint8_t mem[];
        
        // has nr_centroids*3 elements
        float *local_next_centroids = (float*) mem;
        // has nr_centroids elements
        unsigned *local_next_centroid_counts = (unsigned*)(local_next_centroids + nr_centroids*3);

        // zero out block-local data in preparation for next iteration
        for (auto i = threadIdx.x; i < nr_centroids*3; i += blockDim.x)
                local_next_centroids[i] = 0;

        for (auto i = threadIdx.x; i < nr_centroids; i += blockDim.x)
                local_next_centroid_counts[i] = 0;

        // calculate the next set of centroids; paralell over points
        for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
             i < nr_points; i += blockDim.x * gridDim.x) {
                float point_x = points[i];
                float point_y = points[i+nr_points];
                float point_z = points[i+nr_points*2];
                unsigned closest_idx = 0;
                float closest_distance = FLT_MAX;
                        
                // find the centroid closest to this thread's point
                for (auto j = 0u; j < nr_centroids; ++j) {
                        const float dx = centroids[j] - point_x;
                        const float dy = centroids[j+nr_centroids] - point_y;
                        const float dz = centroids[j+nr_centroids*2] - point_z;
                        const float distance = dx*dx + dy*dy + dz+dz;

                        if (distance < closest_distance) {
                                closest_distance = distance;
                                closest_idx = j;
                        }
                }

                atomicAdd(local_next_centroids + closest_idx,
                          point_x);
                atomicAdd(local_next_centroids + nr_centroids + closest_idx,
                          point_y);
                atomicAdd(local_next_centroids + nr_centroids*2 + closest_idx,
                          point_z);
                atomicAdd(local_next_centroid_counts + closest_idx,
                          1);
        }

        // send the local shared memory to global memory
        const unsigned blk_base = blockIdx.x*nr_centroids;
        for (unsigned i = threadIdx.x; i < nr_centroids*3; i += blockDim.x)
                next_centroids[blk_base+i] = local_next_centroids[i];

        for (unsigned i = threadIdx.x; i < nr_centroids; i += blockDim.x)
                next_centroid_counts[blk_base+i] = local_next_centroid_counts[i];
}

__global__ static void
reduce_counts_kernel(const unsigned nr_centroids,
                     unsigned *next_centroid_counts, // size = nr_centroids*nr_point_blocks
                     const unsigned nr_point_blocks)
{
        // first reduce next_centroid_counts into the first nr_centroids slots of
        // that array
        for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
             i < nr_centroids; i += blockDim.x * gridDim.x) {
                unsigned count_accum = 0;
                
                for (auto j = 0u; j < nr_point_blocks; ++j)
                        count_accum += next_centroid_counts[j*nr_centroids+i];

                next_centroid_counts[i] = count_accum;
        }
}

__global__ static void
move_centroids_kernel(float *centroids,
                      const unsigned nr_centroids,
                      float *next_centroids, // size = nr_centroids*3*nr_blocks
                      unsigned *next_centroid_counts, // size = nr_centroids*nr_point_blocks
                      const unsigned nr_point_blocks) // number of blocks used for the above kernel
{
        // now move the centroids. Since each coordinate is independent, we can
        // do them in parallel
        for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
             i < nr_centroids*3; i += blockDim.x * gridDim.x) {
                unsigned count = next_centroid_counts[i%nr_centroids];
                float accum = 0;

                // reduce the per-block coordinates
                for (auto j = 0u; j < nr_point_blocks; ++j)
                        accum += next_centroids[j*nr_centroids + i];

                // move the centroids
                if (count > 0)
                        centroids[i] = accum/count;
        }
}

static void points_aos_to_soa(const float *aos, std::vector<float>& soa)
{
        auto size = soa.size()/3;
        for (auto i = 0u; i < size; ++i)
                for (auto j = 0u; j < 3; ++j)
                        soa.at(j*size + i) = aos[i*3 + j];
}

static void point_soa_to_aos(const std::vector<float>& soa, float *aos)
{
        auto size = soa.size()/3;
        for (auto i = 0u; i < size; ++i)
                for (auto j = 0u; j < 3; ++j)
                        aos[i*3 + j] = soa.at(j*size + i);
}

void
runGpuTimingTest(const unsigned int numberOfTrials,
                 const unsigned int maxNumberOfBlocks,
                 const unsigned int numberOfThreadsPerBlock,
                 const float * const points_cpu,
                 const unsigned int numberOfPoints,
                 const float * const starting_centroids_cpu,
                 const unsigned int numberOfCentroids,
                 const unsigned int numberOfIterations,
                 float * const finalCentroids_Cpu_AoS,
                 float * const elapsedTime) {

        // calculate the number of blocks
        const unsigned int numberOfBlocksForPoints =
                min(maxNumberOfBlocks,
                    (unsigned int)ceil(numberOfPoints/float(numberOfThreadsPerBlock)));

        // all this work, and it's honestly just like 2
        const unsigned int numberOfBlocksForCentroids =
                min(maxNumberOfBlocks,
                    (unsigned int)ceil(numberOfCentroids/float(numberOfThreadsPerBlock)));

        // prepare soa layout data
        std::vector<float> points_gpu(numberOfPoints*3);
        std::vector<float> starting_centroids_gpu(numberOfCentroids*3);
        points_aos_to_soa(points_cpu, points_gpu);
        points_aos_to_soa(starting_centroids_cpu, starting_centroids_gpu);

        // Allocate device-side points
        dev_mem<float> dev_points{points_gpu};
        dev_mem<float> dev_centroids{starting_centroids_gpu};
        dev_mem<float> dev_nextCentroids{numberOfCentroids * 3 * numberOfBlocksForPoints};
        dev_mem<unsigned> dev_nextCentroidCounts{numberOfCentroids * 3 * numberOfBlocksForPoints};

        *elapsedTime = std::numeric_limits<float>::max();

        // run the test repeatedly
        for (unsigned int trialNumber = 0;
             trialNumber < numberOfTrials; ++trialNumber) {

                // Reset intermediate values for next calculation
                checkCudaError(cudaMemcpy(dev_centroids, starting_centroids_gpu.data(),
                                          numberOfCentroids * 3 * sizeof(float),
                                          cudaMemcpyHostToDevice));

                // this forces the GPU to run another kernel, kind of like
                //  "resetting the cache" for the cpu versions.
                GpuUtilities::resetGpu();

                // Wait for any kernels to stop
                checkCudaError(cudaDeviceSynchronize());

                // Start timing
                auto tic = TimeUtility::getCurrentTime();

                // For each of a fixed number of iterations
                for (auto n = 0u; n < numberOfIterations; ++n) {

                        find_local_centroids_kernel<<<numberOfBlocksForPoints,
                                numberOfThreadsPerBlock,
                                numberOfCentroids*3*sizeof(float)
                                + numberOfCentroids*sizeof(unsigned)
                                                   >>>(dev_points, dev_points.size()/3,
                                                       dev_centroids, dev_centroids.size()/3,
                                                       dev_nextCentroids, dev_nextCentroidCounts);

                        reduce_counts_kernel<<<numberOfBlocksForCentroids,
                                numberOfThreadsPerBlock
                                            >>>(dev_centroids.size()/3,
                                                dev_nextCentroidCounts,
                                                numberOfBlocksForPoints);
                        
                        move_centroids_kernel<<<numberOfBlocksForCentroids,
                                numberOfThreadsPerBlock
                                             >>>(dev_centroids, dev_centroids.size()/3,
                                                 dev_nextCentroids, dev_nextCentroidCounts,
                                                 numberOfBlocksForPoints);

                        // See if there was an error in the kernel launch
                        checkCudaError(cudaPeekAtLastError());
                }

                // Wait for the kernels to stop
                checkCudaError(cudaDeviceSynchronize());

                // Stop timing
                auto toc = TimeUtility::getCurrentTime();
                const float thisTrialsElapsedTime =
                        TimeUtility::getElapsedTime(tic, toc);
                *elapsedTime = std::min(*elapsedTime, thisTrialsElapsedTime);
        }

        // copy device outputs back to host
        std::vector<float> final_centroids_gpu(dev_centroids.size());
        dev_centroids.write_to(final_centroids_gpu);
        point_soa_to_aos(final_centroids_gpu, finalCentroids_Cpu_AoS);
}
