// knn_bvh.cu — compiled by nvcc.
// LBVH-based GPU K-nearest-neighbor search.
//
// Compile with -DKNN_2D for 2-D point clouds (z is ignored / set to 0).

#include "knn_bvh.hpp"
#include <lbvh.cuh>   // found via -I../lbvh  (see Makefile)

#include <algorithm>
#include <climits>
#include <cstdio>
#include <vector>
#include <cmath>
#include <mutex>
#include <stdexcept>

// ---------------------------------------------------------------------------
// AABB getter
// ---------------------------------------------------------------------------
struct Float3AABBGetter {
    __device__ __host__
    lbvh::aabb<float> operator()(const float3& p) const noexcept {
        lbvh::aabb<float> box;
        box.upper = make_float4(p.x, p.y, p.z, 0.f);
        box.lower = make_float4(p.x, p.y, p.z, 0.f);
        return box;
    }
};
using BVH_t = lbvh::bvh<float, float3, Float3AABBGetter>;

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------
struct KNNBVHState {
    BVH_t*    bvh;
    int       n_points;
    int       bvh_capacity;
    int       bvh_capacity_old;
    bool      is_padded;
    float3*   d_queries;
    size_t    d_queries_cap;
    neighbor* d_results;
    size_t    d_results_cap;
    uint32_t* d_stack_pool;
    size_t    d_stack_pool_cap;
    // Cached optimal thread counts per K-tier: 0→K=1, 1→K≤20, 2→K≤32
    int       opt_threads_k[3];
    int*      d_status_flag;
    // Bounding-box volume used to estimate default search radius.
    // For 2-D builds this is the area (z extent = 1).
    float     bbox_volume;
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr int    MAX_K_CAPACITY          = 32;
constexpr int    GLOBAL_STACK_SIZE       = 128;
constexpr double EXTEND_RATIO            = 1.5;
constexpr double GPU_MEMORY_SAFETY_MARGIN = 0.15;

constexpr int STATUS_OK             = 0;
constexpr int STATUS_OVERFLOW       = 1;
constexpr int STATUS_ZERO_NEIGHBORS = 2;

// One BVH is kept alive across destroy/create cycles to avoid expensive
// reallocation when the point cloud changes only modestly.
static KNNBVHState* s_bvh_cached = nullptr;
static std::mutex   s_pool_mutex;

// ---------------------------------------------------------------------------
// Bounding-box volume helper (host, called once per BVH build)
// ---------------------------------------------------------------------------
static float compute_bbox_volume(const float* h_xyz, int n)
{
    if (n <= 0) return 1.f;
    float xmin= h_xyz[0], xmax= h_xyz[0];
    float ymin= h_xyz[1], ymax= h_xyz[1];
    float zmin= h_xyz[2], zmax= h_xyz[2];
    for (int i = 1; i < n; ++i) {
        xmin = fminf(xmin, h_xyz[3*i+0]); xmax = fmaxf(xmax, h_xyz[3*i+0]);
        ymin = fminf(ymin, h_xyz[3*i+1]); ymax = fmaxf(ymax, h_xyz[3*i+1]);
        zmin = fminf(zmin, h_xyz[3*i+2]); zmax = fmaxf(zmax, h_xyz[3*i+2]);
    }
    float dx = xmax - xmin, dy = ymax - ymin, dz = zmax - zmin;
    // Clamp to avoid zero-volume degenerate clouds
    if (dx < 1e-6f) dx = 1e-6f;
    if (dy < 1e-6f) dy = 1e-6f;
    if (dz < 1e-6f) dz = 1e-6f;
#ifdef KNN_2D
    return dx * dy;          // 2-D: area
#else
    return dx * dy * dz;     // 3-D: volume
#endif
}

// Expected squared distance to the K-th nearest neighbor given point density.
//   volume / n_points  = volume per point
//   K * (volume/n)     = volume of sphere containing ~K neighbors
//   r^dim = K*V/n  =>  r^2 = (K*V/n)^(2/dim)
// Multiply by a safety factor so the initial search sphere is generous enough
// to actually contain K neighbors most of the time (avoids costly retries).
static inline float default_radius_sq(float bbox_vol, int n_points, int k)
{
    if (n_points <= 0) return 1e30f;
    float vol_per_k = bbox_vol * k / (float)n_points;
#ifdef KNN_2D
    return powf(vol_per_k, 1.0f) * 4.0f;   // r^2 = (K*A/N) * safety
#else
    return powf(vol_per_k, 2.0f/3.0f) * 4.0f; // r^2 = (K*V/N)^(2/3) * safety
#endif
}

// ---------------------------------------------------------------------------
// Distance helpers
// ---------------------------------------------------------------------------
__device__ __host__
inline float dist2(const float3& a, const float3& b) {
    float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return dx*dx + dy*dy + dz*dz;
}

__device__
inline float point_aabb_dist2(const float3& p, const lbvh::aabb<float>& box) {
    float dx = fmaxf(0.f, fmaxf(box.lower.x - p.x, p.x - box.upper.x));
    float dy = fmaxf(0.f, fmaxf(box.lower.y - p.y, p.y - box.upper.y));
    float dz = fmaxf(0.f, fmaxf(box.lower.z - p.z, p.z - box.upper.z));
    return dx*dx + dy*dy + dz*dz;
}

// ---------------------------------------------------------------------------
// KNN kernel
// Template parameter MAX_K must be >= actual_k; three specialisations below.
// ---------------------------------------------------------------------------
template<int MAX_K>
__global__
void knn_kernel(
    const lbvh::bvh_device<float, float3> bvh,
    const float3* __restrict__  queries,
    uint32_t*    __restrict__  global_stacks,
    const float* __restrict__  guess_radii_sq,
    float                      radius_scale,
    float                      default_r2,
    int                        num_queries,
    int                        actual_k,
    neighbor*    __restrict__  out_results,
    int*         __restrict__  status_flag)
{
    const int qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= num_queries) return;

    const float3 q = queries[qid];
    float cand_dist[MAX_K];
    int   cand_idx [MAX_K];
    int   num_found = 0;

    float current_max_dist2 =
        guess_radii_sq ? (guess_radii_sq[qid] * radius_scale) : default_r2;
    bool need_retry = false;

    do {
        need_retry       = false;
        bool stack_overflow = false;
        num_found        = 0;
        float max_dist2  = current_max_dist2;

        // Two-tier traversal stack
        constexpr int REG_STACK_SIZE = 16;
        uint32_t reg_stack[REG_STACK_SIZE];
        int reg_ptr = 0;

        uint32_t* global_stack = &global_stacks[(size_t)qid * GLOBAL_STACK_SIZE];
        int global_ptr = 0;

        reg_stack[reg_ptr++] = 0;  // push root

        #define PUSH_NODE(idx) do { \
            if (!stack_overflow) { \
                if (reg_ptr < REG_STACK_SIZE || global_ptr <= GLOBAL_STACK_SIZE - REG_STACK_SIZE) { \
                    if (reg_ptr == REG_STACK_SIZE) { \
                        for(int _i = 0; _i < REG_STACK_SIZE; ++_i) \
                            global_stack[global_ptr++] = reg_stack[_i]; \
                        reg_ptr = 0; \
                    } \
                    reg_stack[reg_ptr++] = (idx); \
                } else { \
                    stack_overflow = true; \
                } \
            } \
        } while(0)

        while (!stack_overflow && (reg_ptr > 0 || global_ptr > 0)) {
            if (reg_ptr == 0) {
                for (int i = REG_STACK_SIZE - 1; i >= 0; --i)
                    reg_stack[i] = global_stack[--global_ptr];
                reg_ptr = REG_STACK_SIZE;
            }

            const uint32_t node_idx = reg_stack[--reg_ptr];
            if (point_aabb_dist2(q, bvh.aabbs[node_idx]) > max_dist2) continue;

            const auto& node = bvh.nodes[node_idx];
            if (node.object_idx != 0xFFFFFFFF) {
                // Leaf node
                const float3& p = bvh.objects[node.object_idx];
                float d2 = dist2(q, p);
                if (d2 < max_dist2 || num_found < actual_k) {
                    int ins = (num_found < actual_k) ? num_found++ : actual_k - 1;
                    while (ins > 0 && d2 < cand_dist[ins - 1]) {
                        cand_dist[ins] = cand_dist[ins - 1];
                        cand_idx [ins] = cand_idx [ins - 1];
                        --ins;
                    }
                    cand_dist[ins] = d2;
                    cand_idx [ins] = node.object_idx;
                    if (num_found == actual_k) max_dist2 = cand_dist[actual_k - 1];
                }
            } else {
                // Internal node — visit nearer child first
                float dL = point_aabb_dist2(q, bvh.aabbs[node.left_idx]);
                float dR = point_aabb_dist2(q, bvh.aabbs[node.right_idx]);
                if (dL < dR) {
                    if (dR <= max_dist2) PUSH_NODE(node.right_idx);
                    if (dL <= max_dist2) PUSH_NODE(node.left_idx);
                } else {
                    if (dL <= max_dist2) PUSH_NODE(node.left_idx);
                    if (dR <= max_dist2) PUSH_NODE(node.right_idx);
                }
            }
        }
        #undef PUSH_NODE

        if (stack_overflow) {
            atomicMax(status_flag, STATUS_OVERFLOW);
            break;
        }

        if (num_found < actual_k && current_max_dist2 < 1e20f) {
            current_max_dist2 *= 4.0f;
            if (current_max_dist2 < 1e-6f) current_max_dist2 = 1e-6f;
            if (current_max_dist2 > 1e10f) current_max_dist2 = 1e30f;
            need_retry = true;
        }
    } while (need_retry);

    if (num_found == 0) {
        printf(">>> [KNN FATAL] query %d at (%f,%f,%f) found ZERO neighbors!\n",
               qid, q.x, q.y, q.z);
        atomicMax(status_flag, STATUS_ZERO_NEIGHBORS);
    } else if (num_found < actual_k && threadIdx.x == 0 && blockIdx.x == 0) {
        printf(">>> [KNN WARNING] qid=%d found only %d/%d neighbors. Padded with idx=-1.\n",
               qid, num_found, actual_k);
    }

    for (int i = 0; i < actual_k; ++i) {
        size_t base = (size_t)qid * actual_k + i;
        if (i < num_found) {
            out_results[base].idx   = cand_idx[i];
            out_results[base].dist2 = (double)cand_dist[i];
        } else {
            out_results[base].idx   = -1;
            out_results[base].dist2 = 1e30;
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
static KNNBVHState* get_or_init_bvh_state()
{
    KNNBVHState* s;
    { std::lock_guard<std::mutex> lk(s_pool_mutex); s = s_bvh_cached; s_bvh_cached = nullptr; }
    if (!s) {
        s = new KNNBVHState();
        s->bvh = nullptr; s->bvh_capacity = 0; s->bvh_capacity_old = 0;
        s->is_padded = false;
        s->d_queries = nullptr; s->d_queries_cap = 0;
        s->d_results = nullptr; s->d_results_cap = 0;
        s->d_stack_pool = nullptr; s->d_stack_pool_cap = 0;
        s->d_status_flag = nullptr;
        cudaMalloc(&s->d_status_flag, sizeof(int));
        memset(s->opt_threads_k, 0, sizeof(s->opt_threads_k));
        int minGrid;
        cudaOccupancyMaxPotentialBlockSize(&minGrid, &s->opt_threads_k[0], knn_kernel<1>,  0, 0);
        cudaOccupancyMaxPotentialBlockSize(&minGrid, &s->opt_threads_k[1], knn_kernel<20>, 0, 0);
        cudaOccupancyMaxPotentialBlockSize(&minGrid, &s->opt_threads_k[2], knn_kernel<32>, 0, 0);
    }
    return s;
}

static std::vector<float3> prepare_bvh_points(const float* h_xyz, int n, int cap)
{
    std::vector<float3> pts(cap);
    for (int i = 0; i < n;   ++i) pts[i] = make_float3(h_xyz[3*i], h_xyz[3*i+1], h_xyz[3*i+2]);
    for (int i = n; i < cap; ++i) pts[i] = make_float3(1e30f, 1e30f, 1e30f);  // ghost points
    return pts;
}

static size_t bpbvh_for_n(size_t n)
{
    return n * sizeof(float3)    // objects
         + (2*n-1) * 32          // aabbs  (lbvh::aabb<float> = 32 B)
         + (2*n-1) * 16          // nodes  (lbvh::node       = 16 B)
         + n * 4                 // morton codes
         + n * 4                 // sorted indices
         + n * 8                 // morton64
         + (n-1) * 4;            // flag container
}

static inline size_t bpq_for_k(int k)
{
    return sizeof(float3)
         + (size_t)k * sizeof(neighbor)
         + GLOBAL_STACK_SIZE * sizeof(uint32_t);
}

static void ensure_device_buffers(KNNBVHState* s, int nquery, int k,
                                   bool chk_q, bool chk_r, bool chk_s)
{
    if (nquery <= 0) return;

    size_t req_r = (size_t)nquery * k;
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    bool need_q = chk_q && ((size_t)nquery > s->d_queries_cap);
    bool need_r = chk_r && (req_r           > s->d_results_cap);
    bool need_s = chk_s && ((size_t)nquery > s->d_stack_pool_cap);

    bool force_shrink = false;
    if (free_mem < (size_t)(total_mem * GPU_MEMORY_SAFETY_MARGIN)) {
        bool over_q = chk_q && s->d_queries_cap    > (size_t)(nquery * 1.05);
        bool over_r = chk_r && s->d_results_cap    > (size_t)(req_r  * 1.05);
        bool over_s = chk_s && s->d_stack_pool_cap > (size_t)(nquery * 1.05);
        if (over_q || over_r || over_s) {
            force_shrink = true;
            if (chk_q) need_q = true;
            if (chk_r) need_r = true;
            if (chk_s) need_s = true;
        }
    }
    if (!need_q && !need_r && !need_s) return;

    size_t tq = 0, tr = 0;
    if (force_shrink) {
        tq = (size_t)nquery; tr = req_r;
        printf(">>> [Memory Alert] GPU free < %.0f%%. Shrinking KNN buffers (nquery=%d).\n",
               GPU_MEMORY_SAFETY_MARGIN*100., nquery);
    } else {
        tq = (size_t)(nquery * EXTEND_RATIO);
        tr = tq * k;

        size_t cur_bytes = s->d_queries_cap * sizeof(float3)
                         + s->d_results_cap * sizeof(neighbor)
                         + s->d_stack_pool_cap * GLOBAL_STACK_SIZE * sizeof(uint32_t);
        size_t base_used = (total_mem - free_mem) - cur_bytes;
        size_t max_allow = (size_t)(total_mem * (1.0 - GPU_MEMORY_SAFETY_MARGIN));

        size_t fut_bytes = (need_q ? tq : s->d_queries_cap) * sizeof(float3)
                         + (need_r ? tr : s->d_results_cap) * sizeof(neighbor)
                         + (need_s ? tq : s->d_stack_pool_cap) * GLOBAL_STACK_SIZE * sizeof(uint32_t);

        if (base_used + fut_bytes > max_allow) {
            size_t fixed = (!need_q ? s->d_queries_cap*sizeof(float3) : 0)
                         + (!need_r ? s->d_results_cap*sizeof(neighbor) : 0)
                         + (!need_s ? s->d_stack_pool_cap*GLOBAL_STACK_SIZE*sizeof(uint32_t) : 0);
            size_t budget = (max_allow > base_used + fixed) ? max_allow - base_used - fixed : 0;
            size_t bpq = (need_q ? sizeof(float3) : 0)
                       + (need_r ? (size_t)k * sizeof(neighbor) : 0)
                       + (need_s ? GLOBAL_STACK_SIZE * sizeof(uint32_t) : 0);
            size_t max_cap = bpq ? budget / bpq : 0;
            tq = (max_cap >= (size_t)nquery) ? max_cap : (size_t)nquery;
            tr = tq * k;
        }
    }

    auto alloc = [](void** ptr, size_t bytes, const char* label) {
        cudaFree(*ptr);
        *ptr = nullptr;
        if (bytes == 0) return (size_t)0;
        if (cudaMalloc(ptr, bytes) != cudaSuccess)
            throw std::runtime_error(std::string("cudaMalloc failed for ") + label);
        return bytes / sizeof(char); // dummy
    };

    if (need_q) { cudaFree(s->d_queries); s->d_queries = nullptr;
                  if (tq > 0 && cudaMalloc(&s->d_queries, tq*sizeof(float3)) != cudaSuccess)
                      throw std::runtime_error("cudaMalloc failed for d_queries");
                  s->d_queries_cap = tq; }

    if (need_r) { cudaFree(s->d_results); s->d_results = nullptr;
                  if (tr > 0 && cudaMalloc(&s->d_results, tr*sizeof(neighbor)) != cudaSuccess)
                      throw std::runtime_error("cudaMalloc failed for d_results");
                  s->d_results_cap = tr; }

    if (need_s) { cudaFree(s->d_stack_pool); s->d_stack_pool = nullptr;
                  if (tq > 0 && cudaMalloc(&s->d_stack_pool, tq*GLOBAL_STACK_SIZE*sizeof(uint32_t)) != cudaSuccess)
                      throw std::runtime_error("cudaMalloc failed for d_stack_pool");
                  s->d_stack_pool_cap = tq; }
}

static void launch_knn_kernel(KNNBVHState* s, int nquery, int k,
                               const float* guess_radii_sq)
{
    float radius_scale = 1.0f;
    if (guess_radii_sq) {
#ifdef KNN_2D
        radius_scale = (float)k * 1.5f;
#else
        radius_scale = std::pow((float)k, 2.0f/3.0f) * 1.5f;
#endif
    }

    int tier    = (k <= 1) ? 0 : (k <= 20) ? 1 : 2;
    int threads = s->opt_threads_k[tier];
    if (threads <= 0) threads = 256;
    int blocks  = (nquery - 1) / threads + 1;
    auto bvh_dev = s->bvh->get_device_repr();

    // Default initial search radius from point density (used when no per-query
    // hint is provided).  This is the key to avoiding full-tree traversal.
    float def_r2 = default_radius_sq(s->bbox_volume, s->n_points, k);

    cudaMemset(s->d_status_flag, STATUS_OK, sizeof(int));

    switch (tier) {
        case 0: knn_kernel<1> <<<blocks,threads>>>(bvh_dev, s->d_queries, s->d_stack_pool, guess_radii_sq, radius_scale, def_r2, nquery, k, s->d_results, s->d_status_flag); break;
        case 1: knn_kernel<20><<<blocks,threads>>>(bvh_dev, s->d_queries, s->d_stack_pool, guess_radii_sq, radius_scale, def_r2, nquery, k, s->d_results, s->d_status_flag); break;
        default:knn_kernel<32><<<blocks,threads>>>(bvh_dev, s->d_queries, s->d_stack_pool, guess_radii_sq, radius_scale, def_r2, nquery, k, s->d_results, s->d_status_flag); break;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("knn_kernel launch: ") + cudaGetErrorString(err));

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("knn_kernel execution: ") + cudaGetErrorString(err));

    int h_status = STATUS_OK;
    cudaMemcpy(&h_status, s->d_status_flag, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_status == STATUS_OVERFLOW)
        printf(">>> [KNN WARNING] Stack overflow. Increase GLOBAL_STACK_SIZE (%d).\n", GLOBAL_STACK_SIZE);
    else if (h_status == STATUS_ZERO_NEIGHBORS)
        throw std::runtime_error("KNN: one or more queries found zero neighbors");
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
KNNBVHState* knn_bvh_create(const float* h_xyz, int n_points)
{
    KNNBVHState* s = get_or_init_bvh_state();
    s->n_points = n_points; s->is_padded = false;
    s->bbox_volume = compute_bbox_volume(h_xyz, n_points);
    auto pts = prepare_bvh_points(h_xyz, n_points, n_points);
    if (s->bvh && s->bvh_capacity == n_points) {
        s->bvh->assign(pts.begin(), pts.end());
    } else {
        delete s->bvh;
        s->bvh_capacity = n_points;
        s->bvh = new BVH_t(pts.begin(), pts.end());
    }
    return s;
}

KNNBVHState* knn_bvh_create_padded(const float* h_xyz, int n_points, int req_capacity)
{
    if (req_capacity < n_points) req_capacity = n_points;
    KNNBVHState* s = get_or_init_bvh_state();
    s->n_points = n_points; s->is_padded = true;
    s->bbox_volume = compute_bbox_volume(h_xyz, n_points);

    if (s->bvh && s->bvh_capacity > req_capacity * 3) { delete s->bvh; s->bvh = nullptr; }

    if (!s->bvh || n_points > s->bvh_capacity) {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        size_t cur_used  = s->bvh_capacity ? bpbvh_for_n(s->bvh_capacity) : 0;
        size_t true_avail = free_mem + cur_used;
        size_t target_cap = req_capacity;
        if (bpbvh_for_n(target_cap) > (size_t)(true_avail * (1.0 - GPU_MEMORY_SAFETY_MARGIN))) {
            target_cap = n_points;
            printf(">>> [Warning] BVH VRAM limit hit. Abandoning %.1fx expansion.\n",
                   (double)req_capacity / n_points);
        }
        delete s->bvh; s->bvh = nullptr;
        s->bvh_capacity_old = s->bvh_capacity;
        s->bvh_capacity     = target_cap;
    }

    auto pts = prepare_bvh_points(h_xyz, n_points, s->bvh_capacity);
    if (s->bvh) {
        s->bvh->assign(pts.begin(), pts.end());
    } else {
        if (s->bvh_capacity_old)
            printf("      knn_bvh_create_padded: capacity %d → %d\n",
                   s->bvh_capacity_old, s->bvh_capacity);
        s->bvh = new BVH_t(pts.begin(), pts.end());
    }
    return s;
}

void knn_bvh_destroy(KNNBVHState* s)
{
    if (!s) return;
    KNNBVHState* evict;
    { std::lock_guard<std::mutex> lk(s_pool_mutex); evict = s_bvh_cached; s_bvh_cached = s; }
    if (evict) {
        delete evict->bvh;
        cudaFree(evict->d_queries);
        cudaFree(evict->d_results);
        cudaFree(evict->d_stack_pool);
        cudaFree(evict->d_status_flag);
        delete evict;
    }
}

void* knn_bvh_search(KNNBVHState* s,
                     const float* h_xyz_queries, int nquery, int k,
                     void* h_results, bool sync_to_host,
                     const float* d_guess_radii_sq)
{
    if (k > MAX_K_CAPACITY) {
        char msg[128]; snprintf(msg, sizeof(msg),
            "K=%d exceeds MAX_K_CAPACITY=%d", k, MAX_K_CAPACITY);
        throw std::invalid_argument(msg);
    }
    ensure_device_buffers(s, nquery, k, true, true, true);
    cudaMemcpy(s->d_queries, h_xyz_queries, (size_t)nquery * sizeof(float3),
               cudaMemcpyHostToDevice);
    launch_knn_kernel(s, nquery, k, d_guess_radii_sq);
    if (sync_to_host) {
        cudaMemcpy(h_results, s->d_results,
                   (size_t)nquery * k * sizeof(neighbor), cudaMemcpyDeviceToHost);
        return h_results;
    }
    return s->d_results;
}

float3* knn_bvh_ensure_queries(KNNBVHState* s, int nquery)
{
    ensure_device_buffers(s, nquery, 0, true, false, true);
    return s->d_queries;
}

void* knn_bvh_search_prepared(KNNBVHState* s, int nquery, int k,
                               void* h_results, bool sync_to_host,
                               const float* d_guess_radii_sq)
{
    if (k > MAX_K_CAPACITY) return nullptr;
    ensure_device_buffers(s, nquery, k, false, true, true);
    launch_knn_kernel(s, nquery, k, d_guess_radii_sq);
    if (sync_to_host) {
        cudaMemcpy(h_results, s->d_results,
                   (size_t)nquery * k * sizeof(neighbor), cudaMemcpyDeviceToHost);
        return h_results;
    }
    return s->d_results;
}

void knn_bvh_mem_info(size_t* free_bytes, size_t* total_bytes)
{
    cudaMemGetInfo(free_bytes, total_bytes);
}

int knn_bvh_max_batch(KNNBVHState* s, int k)
{
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t alloc = 0;
    if (s) {
        alloc += s->d_queries_cap    * sizeof(float3);
        alloc += s->d_results_cap    * sizeof(neighbor);
        alloc += s->d_stack_pool_cap * GLOBAL_STACK_SIZE * sizeof(uint32_t);
        if (s->bvh_capacity) alloc += bpbvh_for_n(s->bvh_capacity);
    }
    size_t budget = (size_t)((free_mem + alloc) * 0.6);
    size_t max_q  = budget / bpq_for_k(k);
    constexpr size_t MAX_SAFE = (size_t)INT_MAX - 1023;
    return (int)std::min(max_q / 16, MAX_SAFE);
}

