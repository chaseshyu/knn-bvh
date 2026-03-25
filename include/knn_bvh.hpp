#pragma once
#include <cstddef>

// Compatibility shims so this header compiles cleanly with g++ / any C++
// compiler, not just nvcc.  nvcc defines __CUDACC__; g++ does not.
#ifndef __CUDACC__
#  ifndef __host__
#    define __host__
#  endif
#  ifndef __device__
#    define __device__
#  endif
// float3 is a CUDA built-in; provide a plain-C++ equivalent.
struct float3 { float x, y, z; };
inline float3 make_float3(float x, float y, float z) { return {x, y, z}; }
#endif

// ---------------------------------------------------------------------------
// knn_bvh — GPU K-Nearest Neighbors via LBVH
//
// Build a BVH on the GPU from a set of 3-D (or 2-D) points, then query it
// for the K nearest neighbors of any set of query points.
//
// Points are always represented as packed float triples (x, y, z).
// For 2-D problems compile with -DKNN_2D; the z component is treated as 0.
// ---------------------------------------------------------------------------

// Result type returned for every (query, neighbor) pair.
struct neighbor {
    int    idx;    // index into the source point array (-1 = not found)
    double dist2;  // squared Euclidean distance

    __host__ __device__ neighbor() : idx(-1), dist2(0.0) {}
    __host__ __device__ neighbor(int e, double a) : idx(e), dist2(a) {}
};

// Opaque library state (one per point-cloud).
struct KNNBVHState;

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

// Build a BVH from n_points source points (packed float xyz, length n_points*3).
KNNBVHState* knn_bvh_create(const float* h_xyz, int n_points);

// Build a BVH with capacity >= req_capacity, padding extra slots with ghost
// points. Useful when the point cloud grows over time (avoids reallocation).
KNNBVHState* knn_bvh_create_padded(const float* h_xyz, int n_points, int req_capacity);

// Free BVH and all associated device memory.
void knn_bvh_destroy(KNNBVHState* state);

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

// Perform KNN search.
//   h_xyz_queries    – packed float xyz, length nquery * 3 (host pointer)
//   nquery           – number of query points
//   k                – number of neighbors per query (max 32)
//   h_results        – caller-owned host buffer of size nquery * k;
//                      used when sync_to_host = true
//   sync_to_host     – true  -> D2H copy, return h_results
//                      false -> skip copy, return device pointer to results
//   d_guess_radii_sq – (optional) device pointer to nquery floats with
//                      per-query estimated squared search radius for
//                      early pruning; pass nullptr to disable
//
// Returns h_results when sync_to_host=true, device pointer otherwise.
void* knn_bvh_search(KNNBVHState* state,
                     const float* h_xyz_queries, int nquery, int k,
                     void* h_results, bool sync_to_host,
                     const float* d_guess_radii_sq = nullptr);

// ---------------------------------------------------------------------------
// Memory helpers
// ---------------------------------------------------------------------------

// Recommended maximum batch size for a single knn_bvh_search call,
// based on currently available GPU memory.
int knn_bvh_max_batch(KNNBVHState* state, int k);

// Query current free and total GPU memory in bytes.
void knn_bvh_mem_info(size_t* free_bytes, size_t* total_bytes);

// ---------------------------------------------------------------------------
// Advanced: GPU-direct query path (avoids host-to-device copy)
// ---------------------------------------------------------------------------

// Ensure the internal d_queries buffer has space for >= nquery float3 entries.
// Returns the raw device pointer; write your query points directly into it
// (e.g. via a CUDA kernel or cudaMemcpy).
float3* knn_bvh_ensure_queries(KNNBVHState* state, int nquery);

// Like knn_bvh_search but assumes d_queries has already been populated via
// knn_bvh_ensure_queries(); skips the host-to-device copy.
void* knn_bvh_search_prepared(KNNBVHState* state, int nquery, int k,
                               void* h_results, bool sync_to_host,
                               const float* d_guess_radii_sq = nullptr);
