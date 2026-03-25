// examples/cxx_usage.cpp
// Benchmark / smoke-test for knn-bvh from plain C++ (no CUDA syntax).
//
// When compiled with nvc++ -acc=gpu -cuda -DACC (openacc=1 in the Makefile),
// the brute-force KNN check and point-perturbation loop are offloaded to the
// GPU via OpenACC, mirroring the pattern used in DynEarthSol3D.
// Without OpenACC the brute-force falls back to OpenMP and the perturbation
// runs on a single thread.
//
// Usage: cxx_usage [N_PTS] [N_QUERY] [N_REBUILD]
//        cxx_usage 500000 100000 3

#include "knn_bvh.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static double ms_since(Clock::time_point t0)
{ return Ms(Clock::now() - t0).count(); }

// ---------------------------------------------------------------------------
// LCG random float in [0, 1)  — CPU only (sequential state)
// ---------------------------------------------------------------------------
static float lcg(unsigned int& s)
{
    s = s * 1664525u + 1013904223u;
    return (s >> 8) * (1.f / (1u << 24));
}

static std::vector<float> rand_pts(int n, unsigned int seed = 0)
{
    std::vector<float> v(n * 3);
    unsigned int s = seed;
    for (auto& x : v) x = lcg(s);
    return v;
}

// ---------------------------------------------------------------------------
// Max-heap insert for GPU brute-force KNN
//   heap[0] always holds the current k-th nearest (farthest of the top-k).
//   Inserting a closer point replaces heap[0] and sifts down.
// ---------------------------------------------------------------------------
#pragma acc routine seq
static void knn_insert(neighbor* heap, int k, int idx, double d2)
{
    if (d2 >= heap[0].dist2) return;
    heap[0] = {idx, d2};
    for (int i = 0;;) {
        int l = 2*i+1, r = 2*i+2, m = i;
        if (l < k && heap[l].dist2 > heap[m].dist2) m = l;
        if (r < k && heap[r].dist2 > heap[m].dist2) m = r;
        if (m == i) break;
        neighbor t = heap[i]; heap[i] = heap[m]; heap[m] = t;
        i = m;
    }
}

// ---------------------------------------------------------------------------
// Brute-force KNN — CPU (OpenMP) or GPU (OpenACC)
// ---------------------------------------------------------------------------
static std::vector<neighbor> brute_force(const float* pts, int np,
                                          const float* qrs, int nq, int k)
{
    std::vector<neighbor> res(nq * k);
    neighbor* r = res.data();

#ifndef ACC
    // CPU path: per-thread temporary buffer + std::partial_sort
    #pragma omp parallel for schedule(dynamic, 64)
    for (int qi = 0; qi < nq; ++qi) {
        std::vector<std::pair<float,int>> row(np);
        float qx = qrs[3*qi], qy = qrs[3*qi+1], qz = qrs[3*qi+2];
        for (int i = 0; i < np; ++i) {
            float dx = pts[3*i]-qx, dy = pts[3*i+1]-qy, dz = pts[3*i+2]-qz;
            row[i] = { dx*dx + dy*dy + dz*dz, i };
        }
        std::partial_sort(row.begin(), row.begin()+k, row.end());
        for (int j = 0; j < k; ++j)
            r[qi*k+j] = { row[j].second, (double)row[j].first };
    }
#else
    // GPU path: each query runs independently on a gang/vector thread.
    // knn_insert maintains a max-heap of k candidates per query.
    // With -gpu=mem:managed all pointers are accessible on the device.
    #pragma acc parallel loop gang vector async
    for (int qi = 0; qi < nq; ++qi) {
        neighbor* h = r + qi * k;
        for (int j = 0; j < k; ++j) h[j] = {-1, 1e30};
        float qx = qrs[3*qi], qy = qrs[3*qi+1], qz = qrs[3*qi+2];
        for (int i = 0; i < np; ++i) {
            float dx = pts[3*i]-qx, dy = pts[3*i+1]-qy, dz = pts[3*i+2]-qz;
            knn_insert(h, k, i, (double)(dx*dx + dy*dy + dz*dz));
        }
    }
    #pragma acc wait
    // sort each query's heap ascending (nearest first)
    for (int qi = 0; qi < nq; ++qi)
        std::sort(r + qi*k, r + (qi+1)*k,
                  [](const neighbor& a, const neighbor& b){ return a.dist2 < b.dist2; });
#endif
    return res;
}

static int verify(const neighbor* gpu, const neighbor* ref,
                  int nq, int k, float tol = 1e-3f)
{
    int errs = 0;
    for (int qi = 0; qi < nq; ++qi) {
        double gd = gpu[qi*k].dist2, rd = ref[qi*k].dist2;
        double rel = (rd > 1e-12) ? std::abs(gd-rd)/rd : std::abs(gd-rd);
        if (rel > tol) {
            if (errs < 5)
                printf("  [MISMATCH] qi=%d  gpu_dist2=%.8f  ref_dist2=%.8f  rel=%.4f\n",
                       qi, gd, rd, (float)rel);
            ++errs;
        }
    }
    return errs;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int N_PTS    = (argc > 1) ? atoi(argv[1]) : 500000;
    int N_QUERY  = (argc > 2) ? atoi(argv[2]) : 100000;
    int N_REBUILD= (argc > 3) ? atoi(argv[3]) : 3;

    if (N_PTS < 1 || N_QUERY < 1 || N_REBUILD < 1) {
        fprintf(stderr, "Usage: %s [N_PTS] [N_QUERY] [N_REBUILD]\n", argv[0]);
        return 1;
    }

    // ------------------------------------------------------------------
    // Print configuration
    // ------------------------------------------------------------------
    {
        size_t free_b, total_b;
        knn_bvh_mem_info(&free_b, &total_b);
        printf("=== knn-bvh benchmark (C++ host");
#ifdef ACC
        printf(", OpenACC GPU");
#endif
        printf(") ===\n");
        printf("  source points : %d (%.1f M)\n", N_PTS,   N_PTS  /1e6);
        printf("  query  points : %d (%.1f M)\n", N_QUERY, N_QUERY/1e6);
        printf("  BVH rebuilds  : %d\n",           N_REBUILD);
        printf("  GPU memory    : %.1f / %.1f GiB\n",
               free_b/1073741824.0, total_b/1073741824.0);
        printf("\n");
    }

    // ------------------------------------------------------------------
    // Generate data (flat float xyz arrays)
    // ------------------------------------------------------------------
    printf("[1] Generating point cloud (%d pts)... ", N_PTS); fflush(stdout);
    auto t0  = Clock::now();
    auto pts = rand_pts(N_PTS,  42);
    auto qrs = rand_pts(N_QUERY, 99);
    printf("%.0f ms\n", ms_since(t0));

    // ------------------------------------------------------------------
    // Build BVH (padded to 2x for rebuild test)
    // ------------------------------------------------------------------
    printf("[2] Building padded BVH (capacity = 2x)... "); fflush(stdout);
    t0 = Clock::now();
    KNNBVHState* bvh = knn_bvh_create_padded(pts.data(), N_PTS, 2*N_PTS);
    double build_ms = ms_since(t0);
    printf("%.0f ms\n", build_ms);

    // ------------------------------------------------------------------
    // K sweep
    // ------------------------------------------------------------------
    const int KS[] = { 1, 4, 8, 16, 32 };
    const int NK   = (int)(sizeof(KS)/sizeof(KS[0]));

    printf("\n[3] Search throughput (K sweep, N_QUERY=%d)\n", N_QUERY);
    printf("    %-4s  %10s  %12s\n", "K", "time(ms)", "throughput");
    printf("    %-4s  %10s  %12s\n", "---", "--------", "----------");

    std::vector<neighbor> results;
    for (int ki = 0; ki < NK; ++ki) {
        int k     = KS[ki];
        int batch = std::min(N_QUERY, knn_bvh_max_batch(bvh, k));

        results.resize((size_t)N_QUERY * k);

        // Warm-up
        knn_bvh_search(bvh, qrs.data(), std::min(N_QUERY, 1024), k,
                       results.data(), true);

        const int REPS = 3;
        auto tb = Clock::now();
        for (int r = 0; r < REPS; ++r) {
            int done = 0;
            while (done < N_QUERY) {
                int cur = std::min(batch, N_QUERY - done);
                knn_bvh_search(bvh, qrs.data() + done*3, cur, k,
                               results.data() + (size_t)done*k, true);
                done += cur;
            }
        }
        double ms   = ms_since(tb) / REPS;
        double mqps = (N_QUERY / ms) * 1e-3;
        printf("    %-4d  %10.1f  %9.2f Mq/s\n", k, ms, mqps);
    }

    // ------------------------------------------------------------------
    // Accuracy check (brute-force on a small subset)
    // ------------------------------------------------------------------
    const int N_VERIFY = 500;
    const int K_VERIFY = 4;

    printf("\n[4] Accuracy check  (brute-force vs GPU, %d queries, K=%d)\n",
           N_VERIFY, K_VERIFY);

    std::vector<neighbor> gpu_res((size_t)N_VERIFY * K_VERIFY);

    // sync_to_host=false: results stay in the library's device buffer (d_neighbors
    // is a raw cudaMalloc pointer, i.e. s->d_results).  In non-ACC builds
    // sync_to_host=true copies them to the host directly; gpu_res.data() is then
    // used as d_neighbors so the copy loop below becomes a no-op memcpy.
    // Mirrors the DynEarthSol3D nn-interpolation.cxx GPU-direct search pattern.
#ifdef ACC
    neighbor* d_neighbors = static_cast<neighbor*>(
        knn_bvh_search(bvh, qrs.data(), N_VERIFY, K_VERIFY,
                       nullptr, false));
#else
    knn_bvh_search(bvh, qrs.data(), N_VERIFY, K_VERIFY, gpu_res.data(), true);
    neighbor* d_neighbors = gpu_res.data();
#endif

#ifndef ACC
    #pragma omp parallel for
#endif
    #pragma acc parallel loop gang vector deviceptr(d_neighbors)
    for (int i = 0; i < N_VERIFY * K_VERIFY; ++i)
        gpu_res[i] = d_neighbors[i];
#ifdef ACC
    #pragma acc wait
#endif

    auto t_bf   = Clock::now();
    auto cpu_res = brute_force(pts.data(), N_PTS, qrs.data(), N_VERIFY, K_VERIFY);
    printf("    brute-force time: %.0f ms\n", ms_since(t_bf));

    int errs = verify(gpu_res.data(), cpu_res.data(), N_VERIFY, K_VERIFY);
    if (errs == 0)
        printf("    PASS — all %d results match within tolerance.\n", N_VERIFY);
    else
        printf("    FAIL — %d / %d mismatches!\n", errs, N_VERIFY);

    // ------------------------------------------------------------------
    // Rebuild test
    // ------------------------------------------------------------------
    printf("\n[5] In-place BVH rebuild (%d iterations)\n", N_REBUILD);

    const int K_RB = 8;
    results.resize((size_t)N_QUERY * K_RB);

    float* p = pts.data();

    double total_rebuild_ms = 0, total_search_ms = 0;
    for (int r = 0; r < N_REBUILD; ++r) {
        unsigned int seed = (unsigned int)r * 1234567u;

#ifndef ACC
        // CPU: sequential LCG perturbation
        for (int i = 0; i < N_PTS * 3; ++i)
            p[i] += (lcg(seed) - 0.5f) * 0.02f;
#else
        // GPU: per-index hash so each element is independent
        unsigned int s0 = seed;
        #pragma acc parallel loop gang vector async
        for (int i = 0; i < N_PTS * 3; ++i) {
            unsigned int h = (s0 ^ (unsigned int)i) * 2654435761u;
            float rnd = (h >> 8) * (1.f / (1u << 24));
            p[i] += (rnd - 0.5f) * 0.02f;
        }
        #pragma acc wait
#endif

        auto tr = Clock::now();
        bvh = knn_bvh_create_padded(p, N_PTS, 2*N_PTS);
        total_rebuild_ms += ms_since(tr);

        auto ts  = Clock::now();
        int done = 0, batch = std::min(N_QUERY, knn_bvh_max_batch(bvh, K_RB));
        while (done < N_QUERY) {
            int cur = std::min(batch, N_QUERY - done);
            knn_bvh_search(bvh, qrs.data() + done*3, cur, K_RB,
                           results.data() + (size_t)done*K_RB, true);
            done += cur;
        }
        total_search_ms += ms_since(ts);

        printf("    rebuild %d: build=%.0f ms  search(K=%d)=%.0f ms  %.2f Mq/s\n",
               r+1,
               total_rebuild_ms / (r+1),
               K_RB,
               total_search_ms  / (r+1),
               (N_QUERY / (total_search_ms / (r+1))) * 1e-3);
    }

    // ------------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------------
    printf("\n=== Summary ===\n");
    printf("  Points        : %d\n", N_PTS);
    printf("  Queries       : %d\n", N_QUERY);
    printf("  Initial build : %.0f ms\n", build_ms);
    if (N_REBUILD > 0)
        printf("  Avg rebuild   : %.0f ms  (over %d iterations)\n",
               total_rebuild_ms / N_REBUILD, N_REBUILD);
    printf("  Accuracy      : %s\n", errs == 0 ? "PASS" : "FAIL");

    knn_bvh_destroy(bvh);
    return errs > 0 ? 1 : 0;
}
