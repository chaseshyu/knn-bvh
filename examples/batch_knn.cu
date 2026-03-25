// examples/batch_knn.cu
// Demonstrates batched KNN when the query set is larger than GPU memory allows
// in a single call, using knn_bvh_max_batch() to determine safe batch size.
//
// Also shows the GPU-direct path (knn_bvh_ensure_queries /
// knn_bvh_search_prepared) that avoids a host-to-device memcpy when
// queries are already resident on the GPU.
//
// Compile (from repo root):
//   make examples

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include "knn_bvh.hpp"

// Helper: fill a device float3 buffer from a host float[3*n] array via kernel.
__global__ static void fill_queries_kernel(float3* dst,
                                            const float* src_x,
                                            const float* src_y,
                                            const float* src_z,
                                            int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { dst[i].x = src_x[i]; dst[i].y = src_y[i]; dst[i].z = src_z[i]; }
}

int main()
{
    // ------------------------------------------------------------------
    // Point cloud
    // ------------------------------------------------------------------
    const int N_PTS   = 500000;
    const int N_QUERY = 200000;
    const int K       = 8;

    srand(7);
    std::vector<float> pts(N_PTS * 3);
    for (auto& v : pts) v = (float)rand() / RAND_MAX;

    // ------------------------------------------------------------------
    // Build BVH with padding (capacity = 2×N_PTS) so that if the cloud
    // grows slightly later, the BVH can be rebuilt in-place without
    // reallocating device memory.
    // ------------------------------------------------------------------
    KNNBVHState* bvh = knn_bvh_create_padded(pts.data(), N_PTS, 2 * N_PTS);
    printf("BVH built  N_PTS=%d  K=%d  N_QUERY=%d\n", N_PTS, K, N_QUERY);

    // ------------------------------------------------------------------
    // Determine safe batch size
    // ------------------------------------------------------------------
    int batch = knn_bvh_max_batch(bvh, K);
    batch = std::min(batch, N_QUERY);
    printf("Batch size: %d queries per launch\n", batch);

    // ------------------------------------------------------------------
    // Prepare queries
    // ------------------------------------------------------------------
    std::vector<float> qx(N_QUERY), qy(N_QUERY), qz(N_QUERY);
    for (int i = 0; i < N_QUERY; ++i) {
        qx[i] = (float)rand() / RAND_MAX;
        qy[i] = (float)rand() / RAND_MAX;
        qz[i] = (float)rand() / RAND_MAX;
    }

    std::vector<neighbor> all_results(N_QUERY * K);

    // ------------------------------------------------------------------
    // Batched search — GPU-direct path
    // ------------------------------------------------------------------
    int processed = 0;
    while (processed < N_QUERY) {
        int cur = std::min(batch, N_QUERY - processed);

        // Grab pre-allocated device query buffer
        float3* d_q = knn_bvh_ensure_queries(bvh, cur);

        // Populate device queries from host via a tiny CUDA kernel
        // (in a real application the data might already be on the GPU)
        float *d_x, *d_y, *d_z;
        cudaMalloc(&d_x, cur * sizeof(float));
        cudaMalloc(&d_y, cur * sizeof(float));
        cudaMalloc(&d_z, cur * sizeof(float));
        cudaMemcpy(d_x, qx.data() + processed, cur * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, qy.data() + processed, cur * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_z, qz.data() + processed, cur * sizeof(float), cudaMemcpyHostToDevice);

        int threads = 256, blocks = (cur + threads - 1) / threads;
        fill_queries_kernel<<<blocks, threads>>>(d_q, d_x, d_y, d_z, cur);
        cudaDeviceSynchronize();

        cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);

        // Search — queries already on GPU, skip H2D copy
        knn_bvh_search_prepared(bvh, cur, K,
                                  all_results.data() + processed * K,
                                  /*sync_to_host=*/true);
        processed += cur;
        printf("  processed %d / %d\r", processed, N_QUERY);
        fflush(stdout);
    }
    printf("\nSearch complete.\n");

    // ------------------------------------------------------------------
    // Spot-check: print first 5 query results
    // ------------------------------------------------------------------
    for (int qi = 0; qi < 5; ++qi) {
        printf("Query %2d (%.3f,%.3f,%.3f): ", qi, qx[qi], qy[qi], qz[qi]);
        for (int j = 0; j < K; ++j) {
            const neighbor& nb = all_results[qi * K + j];
            printf("%d(d=%.4f) ", nb.idx, sqrtf((float)nb.dist2));
        }
        printf("\n");
    }

    knn_bvh_destroy(bvh);
    return 0;
}
