// examples/basic_knn.cu
// Build a BVH from a point cloud and query it for K nearest neighbors.
//
// Compile (from repo root):
//   make examples
// or manually:
//   nvcc -std=c++14 -O2 -I include -I lbvh \
//        examples/basic_knn.cu src/knn_bvh.cu -o basic_knn

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "knn_bvh.hpp"

int main()
{
    // ------------------------------------------------------------------
    // 1. Build the source point cloud (random cube, N_PTS points)
    // ------------------------------------------------------------------
    const int N_PTS = 100000;
    const int K     = 5;

    std::vector<float> pts(N_PTS * 3);
    srand(42);
    for (int i = 0; i < N_PTS * 3; ++i)
        pts[i] = (float)rand() / RAND_MAX;   // uniform in [0, 1]^3

    // ------------------------------------------------------------------
    // 2. Build the BVH
    // ------------------------------------------------------------------
    KNNBVHState* bvh = knn_bvh_create(pts.data(), N_PTS);
    printf("BVH built for %d points.\n", N_PTS);

    // ------------------------------------------------------------------
    // 3. Build query points (10 random points)
    // ------------------------------------------------------------------
    const int N_QUERY = 10;
    std::vector<float> queries(N_QUERY * 3);
    for (int i = 0; i < N_QUERY * 3; ++i)
        queries[i] = (float)rand() / RAND_MAX;

    // ------------------------------------------------------------------
    // 4. Allocate host result buffer and search
    // ------------------------------------------------------------------
    std::vector<neighbor> results(N_QUERY * K);

    knn_bvh_search(bvh,
                   queries.data(), N_QUERY, K,
                   results.data(), /*sync_to_host=*/true);

    // ------------------------------------------------------------------
    // 5. Print results
    // ------------------------------------------------------------------
    for (int qi = 0; qi < N_QUERY; ++qi) {
        printf("Query %2d (%.3f, %.3f, %.3f)  ->  top-%d neighbors:\n",
               qi, queries[3*qi], queries[3*qi+1], queries[3*qi+2], K);
        for (int j = 0; j < K; ++j) {
            const neighbor& nb = results[qi * K + j];
            printf("    [%d] idx=%6d  dist=%.6f  pt=(%.3f,%.3f,%.3f)\n",
                   j, nb.idx, sqrtf((float)nb.dist2),
                   pts[3*nb.idx], pts[3*nb.idx+1], pts[3*nb.idx+2]);
        }
    }

    // ------------------------------------------------------------------
    // 6. Destroy
    // ------------------------------------------------------------------
    knn_bvh_destroy(bvh);
    return 0;
}
