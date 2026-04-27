[![DOI](https://zenodo.org/badge/1196690829.svg)](https://doi.org/10.5281/zenodo.19827723)
[![CUDA build](https://github.com/GeoFLAC/knn-bvh/actions/workflows/cuda-build.yml/badge.svg)](https://github.com/GeoFLAC/knn-bvh/actions/workflows/cuda-build.yml)

# knn-bvh

GPU-accelerated K-nearest-neighbor search using a Linear BVH (LBVH) built on CUDA.

Originally developed as the GPU backend of [DynEarthSol3D](https://github.com/GeoFLAC/DynEarthSol),
extracted here as a standalone library.

## Features

- Single `knn_bvh_search()` call handles both small and large query batches
- Automatic GPU memory management with configurable safety margin
- BVH caching across `destroy`/`create` cycles to avoid repeated reallocation
- Padded BVH (`knn_bvh_create_padded`) for point clouds that grow over time
- GPU-direct query path (`knn_bvh_ensure_queries` / `knn_bvh_search_prepared`) for zero-copy when queries are already on the GPU
- Works for both 2-D (`-DKNN_2D`) and 3-D point clouds
- Supports K up to 32

## Dependencies

| Dependency | Version | Notes |
|-----------|---------|-------|
| CUDA Toolkit | ≥ 11.0 | `nvcc` must be on `PATH` |
| [lbvh](https://github.com/ToruNiina/lbvh) | bundled | header-only, in `lbvh/` |

> **lbvh** is bundled in this repository. It implements the algorithm from
> Karras, "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees" (HPG 2012).

## Build

```bash
# Static library (default 3-D, auto-detect GPU architecture)
make

# 2-D point clouds
make NDIM=2

# Explicit GPU architecture (e.g. sm_80 for A100, sm_86 for RTX 3090)
make SM=80

# Examples
make examples

# Debug build
make DEBUG=1 examples
```

Output:
- `lib/libknn_bvh.3d.a` (or `.2d.a`) — static library
- `bin/basic_knn.3d`, `bin/batch_knn.3d` — example programs (suffix matches `NDIM`)

## Quick Start

```cpp
#include "knn_bvh.hpp"
#include <vector>

int main() {
    const int N = 100000, K = 5, NQ = 1000;

    // source points: packed float xyz
    std::vector<float> pts(N * 3);
    // ... fill pts ...

    // build BVH
    KNNBVHState* bvh = knn_bvh_create(pts.data(), N);

    // query points
    std::vector<float> queries(NQ * 3);
    // ... fill queries ...

    // result buffer
    std::vector<neighbor> results(NQ * K);

    knn_bvh_search(bvh,
                   queries.data(), NQ, K,
                   results.data(), /*sync_to_host=*/true);

    for (int i = 0; i < NQ; ++i)
        for (int j = 0; j < K; ++j)
            printf("query %d -> neighbor idx=%d dist=%.4f\n",
                   i, results[i*K+j].idx,
                   sqrtf((float)results[i*K+j].dist2));

    knn_bvh_destroy(bvh);
}
```

Link your code against `lib/libknn_bvh.3d.a` (or `.2d.a`):

```bash
nvcc -std=c++14 -I include -I lbvh my_code.cu -L lib -lknn_bvh.3d -o my_program
```

## API Reference

### Types

```cpp
struct neighbor {
    int    idx;    // index into the source point array (-1 = not found)
    double dist2;  // squared Euclidean distance
};
```

### Build

```cpp
// Build from n_points source points.
// h_xyz: packed float xyz array, length n_points * 3 (host pointer).
KNNBVHState* knn_bvh_create(const float* h_xyz, int n_points);

// Build with pre-allocated capacity (ghost points pad unused slots).
// Use when the point cloud may grow without exceeding req_capacity.
KNNBVHState* knn_bvh_create_padded(const float* h_xyz, int n_points, int req_capacity);

// Destroy and release all device memory.
void knn_bvh_destroy(KNNBVHState* state);
```

### Search

```cpp
// Standard search: copies queries H→D, launches kernel, optionally copies D→H.
// K must be <= 32.
void* knn_bvh_search(KNNBVHState* state,
                     const float* h_xyz_queries, int nquery, int k,
                     void* h_results, bool sync_to_host,
                     const float* d_guess_radii_sq = nullptr);
```

`sync_to_host=true`: results are written to `h_results` and returned.
`sync_to_host=false`: results stay on the GPU; the device pointer is returned.

`d_guess_radii_sq`: optional device array of per-query squared search radii
for early-exit pruning. Pass `nullptr` for exhaustive search.

### Batching

```cpp
// Recommended maximum nquery per single knn_bvh_search call.
int knn_bvh_max_batch(KNNBVHState* state, int k);
```

When `N_QUERY` is large, split into batches of `knn_bvh_max_batch(bvh, K)` or fewer.

### GPU-direct path (advanced)

```cpp
// Ensure device query buffer capacity; returns the device float3* pointer.
float3* knn_bvh_ensure_queries(KNNBVHState* state, int nquery);

// Search assuming d_queries is already populated (skips H→D memcpy).
void* knn_bvh_search_prepared(KNNBVHState* state, int nquery, int k,
                               void* h_results, bool sync_to_host,
                               const float* d_guess_radii_sq = nullptr);
```

### Utilities

```cpp
// Query free/total GPU memory.
void knn_bvh_mem_info(size_t* free_bytes, size_t* total_bytes);
```

## Examples

| File | Description |
|------|-------------|
| `examples/basic_knn.cu` | Build a BVH, search 10 queries, print results |
| `examples/batch_knn.cu` | Large query set split into batches; GPU-direct query path |
| `examples/cxx_usage.cpp` | Call the library from plain C++ (no CUDA syntax); mirrors the DynEarthSol integration pattern |

Run after `make examples`:

```bash
./bin/basic_knn.3d
./bin/batch_knn.3d
# or for 2-D:
# make NDIM=2 examples && ./bin/basic_knn.2d
```

## Using from C++ (no CUDA syntax)

`knn_bvh.hpp` contains only standard C++ types (`float*`, `int`, `size_t`) plus
the `neighbor` POD struct — no CUDA keywords. This means you can include it in
ordinary `.cpp` files and compile them with `g++` or any C++ compiler.

The library is compiled with plain `nvcc -c` (not `-dc`), which embeds the
finalized device binary directly inside the object file. No separate device-link
stub is needed, so the archive links cleanly with both `nvcc` and `g++`.

```bash
# nvcc drives the link so it resolves CUDA library paths automatically;
# g++ (or any C++ compiler) is used as the host compiler.
nvcc -std=c++14 -arch=sm_XX -I include \
     -ccbin g++ -x c++ \
     my_code.cpp \
     -L lib -lknn_bvh.3d \
     -o my_program
```

See `examples/cxx_usage.cpp` for a complete worked example, including a thin
wrapper class that hides the flat `float*` packing — analogous to the `KNN`
class in DynEarthSol's `knn.cxx`.

## Compile-time Options

| Flag | Default | Description |
|------|---------|-------------|
| `KNN_2D` | not set | Treat points as 2-D; z is ignored |

## Notes

- The internal global stack per thread (`GLOBAL_STACK_SIZE = 128`) is sufficient for well-balanced trees. If overflow warnings appear, increase `GLOBAL_STACK_SIZE` in `src/knn_bvh.cu`.
- The library keeps one BVH state alive in a pool across `destroy`/`create` calls, so rebuilding the BVH for a same-size point cloud reuses device allocations.
- Thread safety: `knn_bvh_create` / `knn_bvh_destroy` acquire a mutex for pool access. Search calls are independent and thread-safe as long as each thread uses its own `KNNBVHState*`.

## License

The knn-bvh source code is released under the MIT License (see `LICENSE`).
The bundled `lbvh/` library retains its own license (see `lbvh/LICENSE`).
