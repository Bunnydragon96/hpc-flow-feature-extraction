// CUDA implementation.
//
// Pipeline (all on the GPU):
//   1. flow id   every packet finds or claims a slot of a lock-free hash table;
//                the slot index identifies its flow.
//   2. sort      stable radix sort of the packet indices by slot index, so the
//                packets of a flow become contiguous and stay in time order.
//   3. gather    timestamps and length/flag words are copied into sorted order,
//                flow boundaries are marked and numbered with a prefix sum.
//   4. reduce    the per-flow accumulators are computed. Three variants:
//                  0 "flow"   one thread per flow walks its packets;
//                  1 "atomic" one thread per packet, atomic updates per flow;
//                  2 "chunk"  one thread per fixed-size chunk of the sorted
//                             array, local accumulation, one atomic flush per
//                             flow met in the chunk.
//   5. finalize  statistical features are derived from the accumulators.
#include "flowfeat.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace ff {

#define CK(call)                                                                              \
    do {                                                                                      \
        cudaError_t e_ = (call);                                                              \
        if (e_ != cudaSuccess) {                                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            exit(3);                                                                          \
        }                                                                                     \
    } while (0)

typedef unsigned long long ull;
static const uint32_t EMPTY = 0xFFFFFFFFu;

// ------------------------------------------------------------ kernels
// 1. Flow id. A slot stores the index of the packet that claimed it; keys are
// compared through that packet, so a slot is one 32-bit word and one
// compare-and-swap is enough to insert a 13-byte key.
__global__ void k_flow_id(const Packet* pk, uint32_t n, uint32_t* table, uint32_t mask, uint32_t* fid) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Packet p = pk[i];
    uint32_t s = (uint32_t)pk_hash(p) & mask;
    for (;;) {
        uint32_t cur = table[s];
        if (cur == EMPTY) {
            cur = atomicCAS(&table[s], EMPTY, i);
            if (cur == EMPTY) break;  // this packet now owns the slot
        }
        if (same_flow(pk[cur], p)) break;
        s = (s + 1) & mask;
    }
    fid[i] = s;
}

// 3. Gather packet fields into sorted order and mark the first packet of each flow.
__global__ void k_gather(const Packet* pk, const uint32_t* perm, const uint32_t* fid, uint32_t n,
                         uint32_t* ts_s, uint32_t* meta_s, uint32_t* fx) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t q = perm[i];
    ts_s[i] = pk[q].ts;
    meta_s[i] = pk[q].meta;
    fx[i] = (i == 0 || fid[i] != fid[i - 1]) ? 1u : 0u;
}

// After the prefix sum, fx[i] - 1 is the dense flow number of sorted position i.
__global__ void k_segments(const uint32_t* fx, uint32_t n, uint32_t* seg) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (i == 0 || fx[i] != fx[i - 1]) seg[fx[i] - 1] = i;
}

// Key, packet count and first/last timestamp come from the segment bounds.
__global__ void k_flow_init(const Packet* pk, const uint32_t* perm, const uint32_t* ts_s, const uint32_t* seg,
                            uint32_t nf, FlowRec* R) {
    uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nf) return;
    uint32_t s = seg[f], e = seg[f + 1];
    Packet p = pk[perm[s]];
    FlowRec* r = &R[f];
    r->src = p.src; r->dst = p.dst; r->ports = p.ports; r->proto = pk_proto(p.meta);
    r->n = e - s;
    r->first_ts = ts_s[s];
    r->last_ts = ts_s[e - 1];
    r->len_min = 0xFFFFFFFFu;
    r->iat_min = 0xFFFFFFFFu;
    r->bursts = 1;
}

// 4, variant 0: one thread per flow runs the sequential kernel over its packets.
__global__ void k_reduce_flow(const Packet* pk, const uint32_t* perm, const uint32_t* ts_s,
                              const uint32_t* meta_s, const uint32_t* seg, uint32_t nf, uint32_t gap,
                              FlowRec* R, FlowDerived* D) {
    uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nf) return;
    uint32_t s = seg[f], e = seg[f + 1];
    Packet p = pk[perm[s]];
    FlowRec r;
    rec_init(r, p);
    for (uint32_t i = s + 1; i < e; i++) {
        p.ts = ts_s[i];
        p.meta = meta_s[i];
        rec_update(r, p, gap);
    }
    rec_finish(r);
    R[f] = r;
    D[f] = derive(r);
}

// 4, variant 1: one thread per packet, every contribution is an atomic update.
__global__ void k_reduce_atomic(const uint32_t* ts_s, const uint32_t* meta_s, const uint32_t* fx, uint32_t n,
                                uint32_t gap, FlowRec* R, uint32_t* bs) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    FlowRec* r = &R[fx[i] - 1];
    uint32_t meta = meta_s[i], len = pk_len(meta), fl = pk_flags(meta);
    atomicAdd((ull*)&r->bytes, (ull)len);
    atomicAdd((ull*)&r->sum_len2, (ull)len * len);
    atomicMin(&r->len_min, len);
    atomicMax(&r->len_max, len);
    for (int k = 0; k < NFLAGS; k++)
        if ((fl >> k) & 1u) atomicAdd(&r->flags[k], 1u);
    atomicAdd(&r->hist[len_bin(len)], 1u);
    bool head = (i == 0 || fx[i] != fx[i - 1]);
    uint32_t start = 1;
    if (!head) {
        uint32_t iat = ts_s[i] - ts_s[i - 1];
        atomicMin(&r->iat_min, iat);
        atomicMax(&r->iat_max, iat);
        atomicAdd((ull*)&r->sum_iat2, (ull)iat * iat);
        if (iat >= gap) atomicAdd(&r->bursts, 1u);
        else start = 0;
    }
    bs[i] = start;  // 1 if this packet starts a burst
}

// Local accumulator of the chunked variant. A chunk has at most 255 packets,
// so the 6 flag counters and the 16 histogram counters fit in 8 bits each and
// are packed into three 64-bit registers (no per-thread arrays in memory).
static const uint32_t MAX_CHUNK = 255;
struct Acc {
    ull bytes, sum_len2, sum_iat2, flags, hist_lo, hist_hi;
    uint32_t len_min, len_max, iat_min, iat_max, bursts;
};
__device__ inline void acc_reset(Acc& a) {
    a.bytes = 0; a.sum_len2 = 0; a.sum_iat2 = 0; a.flags = 0; a.hist_lo = 0; a.hist_hi = 0;
    a.len_min = 0xFFFFFFFFu; a.len_max = 0; a.iat_min = 0xFFFFFFFFu; a.iat_max = 0; a.bursts = 0;
}
__device__ inline void acc_flush(const Acc& a, FlowRec* r) {
    atomicAdd((ull*)&r->bytes, a.bytes);
    atomicAdd((ull*)&r->sum_len2, a.sum_len2);
    atomicMin(&r->len_min, a.len_min);
    atomicMax(&r->len_max, a.len_max);
    if (a.iat_min != 0xFFFFFFFFu) {
        atomicAdd((ull*)&r->sum_iat2, a.sum_iat2);
        atomicMin(&r->iat_min, a.iat_min);
        atomicMax(&r->iat_max, a.iat_max);
    }
    if (a.bursts) atomicAdd(&r->bursts, a.bursts);
    for (ull f = a.flags, k = 0; f; f >>= 8, k++)
        if (f & 255u) atomicAdd(&r->flags[k], (uint32_t)(f & 255u));
    for (ull h = a.hist_lo, k = 0; h; h >>= 8, k++)
        if (h & 255u) atomicAdd(&r->hist[k], (uint32_t)(h & 255u));
    for (ull h = a.hist_hi, k = 8; h; h >>= 8, k++)
        if (h & 255u) atomicAdd(&r->hist[k], (uint32_t)(h & 255u));
}

// 4, variant 2: one thread per chunk of `chunk` consecutive sorted packets.
__global__ void k_reduce_chunk(const uint32_t* ts_s, const uint32_t* meta_s, const uint32_t* fx, uint32_t n,
                               uint32_t gap, uint32_t chunk, FlowRec* R, uint32_t* bs) {
    uint64_t c = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t lo64 = c * chunk;
    if (lo64 >= n) return;
    uint32_t lo = (uint32_t)lo64;
    uint32_t hi = (n - lo > chunk) ? lo + chunk : n;
    Acc a;
    acc_reset(a);
    uint32_t cur = fx[lo];
    uint32_t prev_f = lo ? fx[lo - 1] : 0;       // flow numbers start at 1, so 0 never matches
    uint32_t prev_ts = lo ? ts_s[lo - 1] : 0;
    for (uint32_t i = lo; i < hi; i++) {
        uint32_t f = fx[i], ts = ts_s[i], meta = meta_s[i];
        if (f != cur) {
            acc_flush(a, &R[cur - 1]);
            acc_reset(a);
            cur = f;
        }
        uint32_t len = pk_len(meta), fl = pk_flags(meta);
        a.bytes += len;
        a.sum_len2 += (ull)len * len;
        if (len < a.len_min) a.len_min = len;
        if (len > a.len_max) a.len_max = len;
        // spread the 6 flag bits to bit 0 of 6 bytes, then add
        ull spread = 0;
        for (int k = 0; k < NFLAGS; k++) spread |= (ull)((fl >> k) & 1u) << (8 * k);
        a.flags += spread;
        uint32_t bin = len_bin(len);
        if (bin < 8) a.hist_lo += 1ULL << (8 * bin);
        else a.hist_hi += 1ULL << (8 * (bin - 8));
        uint32_t start = 1;
        if (f == prev_f) {
            uint32_t iat = ts - prev_ts;
            if (iat < a.iat_min) a.iat_min = iat;
            if (iat > a.iat_max) a.iat_max = iat;
            a.sum_iat2 += (ull)iat * iat;
            if (iat >= gap) a.bursts++;
            else start = 0;
        }
        bs[i] = start;
        prev_f = f;
        prev_ts = ts;
    }
    acc_flush(a, &R[cur - 1]);
}

// Longest burst. S holds the sorted positions where bursts start; a burst ends
// where the next one starts. Variant 1: one thread per burst.
__global__ void k_burst_atomic(const uint32_t* S, uint32_t nb, uint32_t n, const uint32_t* fx, FlowRec* R) {
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nb) return;
    uint32_t s = S[k], e = (k + 1 < nb) ? S[k + 1] : n;
    atomicMax(&R[fx[s] - 1].max_burst, e - s);
}
// Variant 2: one thread per chunk of consecutive bursts.
__global__ void k_burst_chunk(const uint32_t* S, uint32_t nb, uint32_t n, const uint32_t* fx, uint32_t chunk,
                              FlowRec* R) {
    uint64_t c = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t lo64 = c * chunk;
    if (lo64 >= nb) return;
    uint32_t lo = (uint32_t)lo64;
    uint32_t hi = (nb - lo > chunk) ? lo + chunk : nb;
    uint32_t cur = fx[S[lo]], best = 0;
    for (uint32_t k = lo; k < hi; k++) {
        uint32_t s = S[k], e = (k + 1 < nb) ? S[k + 1] : n;
        uint32_t f = fx[s];
        if (f != cur) {
            atomicMax(&R[cur - 1].max_burst, best);
            cur = f;
            best = 0;
        }
        if (e - s > best) best = e - s;
    }
    atomicMax(&R[cur - 1].max_burst, best);
}

// 5. Finalize (variants 1 and 2).
__global__ void k_finalize(FlowRec* R, FlowDerived* D, uint32_t nf) {
    uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nf) return;
    FlowRec r = R[f];
    rec_finish(r);
    R[f] = r;
    D[f] = derive(r);
}

struct NonZero {
    __host__ __device__ bool operator()(uint32_t x) const { return x != 0; }
};

// ------------------------------------------------------------ buffers
// Device and page-locked host buffers are kept between calls, as a pipeline
// that processes one batch after another would do.
enum { B_PK, B_TABLE, B_FID, B_PERM, B_TS, B_META, B_FX, B_SEG, B_REC, B_DER, B_COUNT };
static void* g_dev[B_COUNT];
static size_t g_cap[B_COUNT];
static void* g_host[3];
static size_t g_hcap[3];

template <class T> static T* dev_buf(int id, size_t count) {
    size_t bytes = count * sizeof(T);
    if (bytes > g_cap[id]) {
        if (g_dev[id]) CK(cudaFree(g_dev[id]));
        g_dev[id] = nullptr;
        cudaError_t e = cudaMalloc(&g_dev[id], bytes);
        if (e != cudaSuccess) {
            fprintf(stderr, "GPU out of memory: buffer %d needs %.1f MiB\n", id, bytes / 1048576.0);
            exit(4);
        }
        g_cap[id] = bytes;
    }
    return (T*)g_dev[id];
}
static void* host_buf(int id, size_t bytes) {
    if (bytes > g_hcap[id]) {
        if (g_host[id]) CK(cudaFreeHost(g_host[id]));
        CK(cudaMallocHost(&g_host[id], bytes));
        g_hcap[id] = bytes;
    }
    return g_host[id];
}

bool gpu_available() {
    int c = 0;
    return cudaGetDeviceCount(&c) == cudaSuccess && c > 0;
}
std::string gpu_name() {
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return "unknown";
    return p.name;
}

static inline uint32_t blocks(size_t items, int tpb) { return (uint32_t)((items + tpb - 1) / tpb); }
static inline double sync_now() {
    CK(cudaDeviceSynchronize());
    return now_s();
}

static Result run_gpu_impl(const Packet* pk_host, size_t n64, uint32_t gap, const GpuOpts& opt);

Result run_gpu(const Packet* pk_host, size_t n64, uint32_t gap, const GpuOpts& opt) {
    try {
        return run_gpu_impl(pk_host, n64, gap, opt);
    } catch (const std::bad_alloc&) {
        fprintf(stderr, "GPU out of memory (in a Thrust temporary buffer): use a smaller batch\n");
        exit(4);
    }
}

static Result run_gpu_impl(const Packet* pk_host, size_t n64, uint32_t gap, const GpuOpts& opt) {
    Result res;
    if (n64 == 0 || n64 > ((size_t)1 << 30)) {
        fprintf(stderr, "run_gpu: between 1 and 2^30 packets per batch\n");
        exit(2);
    }
    const uint32_t n = (uint32_t)n64;
    const int tpb = opt.tpb;
    const uint32_t chunk = opt.chunk < 1 ? 1 : (opt.chunk > (int)MAX_CHUNK ? MAX_CHUNK : (uint32_t)opt.chunk);
    size_t tsize = 1024;
    while (tsize < 2 * (size_t)n) tsize <<= 1;

    Packet* pk = dev_buf<Packet>(B_PK, n);
    uint32_t* table = dev_buf<uint32_t>(B_TABLE, tsize);
    uint32_t* fid = dev_buf<uint32_t>(B_FID, n);
    uint32_t* perm = dev_buf<uint32_t>(B_PERM, n);
    uint32_t* ts_s = dev_buf<uint32_t>(B_TS, n);
    uint32_t* meta_s = dev_buf<uint32_t>(B_META, n);
    uint32_t* fx = dev_buf<uint32_t>(B_FX, n);

    const Packet* src = pk_host;
    if (opt.pinned) {  // stage the input in page-locked memory (not timed: a loader would read into it)
        Packet* h = (Packet*)host_buf(0, (size_t)n * sizeof(Packet));
        memcpy(h, pk_host, (size_t)n * sizeof(Packet));
        src = h;
    }
    CK(cudaDeviceSynchronize());

    double t0 = now_s();
    CK(cudaMemcpy(pk, src, (size_t)n * sizeof(Packet), cudaMemcpyHostToDevice));
    double t1 = sync_now();

    // 1. flow id
    CK(cudaMemset(table, 0xFF, tsize * sizeof(uint32_t)));
    k_flow_id<<<blocks(n, tpb), tpb>>>(pk, n, table, (uint32_t)(tsize - 1), fid);
    double t2 = sync_now();

    // 2. sort packet indices by flow id (stable: time order is kept inside a flow)
    thrust::sequence(thrust::device, perm, perm + n);
    thrust::stable_sort_by_key(thrust::device, fid, fid + n, perm);
    double t3 = sync_now();

    // 3. gather, number the flows, find the segment bounds
    k_gather<<<blocks(n, tpb), tpb>>>(pk, perm, fid, n, ts_s, meta_s, fx);
    thrust::inclusive_scan(thrust::device, fx, fx + n, fx);
    uint32_t nf = 0;
    CK(cudaMemcpy(&nf, fx + (n - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost));
    uint32_t* seg = dev_buf<uint32_t>(B_SEG, (size_t)nf + 1);
    FlowRec* R = dev_buf<FlowRec>(B_REC, nf);
    FlowDerived* D = dev_buf<FlowDerived>(B_DER, nf);
    k_segments<<<blocks(n, tpb), tpb>>>(fx, n, seg);
    CK(cudaMemcpy(seg + nf, &n, sizeof(uint32_t), cudaMemcpyHostToDevice));
    double t4 = sync_now();

    // 4. reduce
    double t5;
    if (opt.variant == 0) {
        k_reduce_flow<<<blocks(nf, tpb), tpb>>>(pk, perm, ts_s, meta_s, seg, nf, gap, R, D);
        t5 = sync_now();
    } else {
        CK(cudaMemset(R, 0, (size_t)nf * sizeof(FlowRec)));
        k_flow_init<<<blocks(nf, tpb), tpb>>>(pk, perm, ts_s, seg, nf, R);
        uint32_t* bs = fid;   // the sorted slot numbers are no longer needed
        if (opt.variant == 1) k_reduce_atomic<<<blocks(n, tpb), tpb>>>(ts_s, meta_s, fx, n, gap, R, bs);
        else k_reduce_chunk<<<blocks(((size_t)n + chunk - 1) / chunk, tpb), tpb>>>(ts_s, meta_s, fx, n, gap, chunk, R, bs);
        CK(cudaDeviceSynchronize());   // k_flow_init must be done before perm is reused
        uint32_t* S = perm;
        uint32_t nb = (uint32_t)(thrust::copy_if(thrust::device, thrust::counting_iterator<uint32_t>(0),
                                                 thrust::counting_iterator<uint32_t>(n), bs, S, NonZero()) - S);
        if (opt.variant == 1) k_burst_atomic<<<blocks(nb, tpb), tpb>>>(S, nb, n, fx, R);
        else k_burst_chunk<<<blocks(((size_t)nb + chunk - 1) / chunk, tpb), tpb>>>(S, nb, n, fx, chunk, R);
        t5 = sync_now();
        k_finalize<<<blocks(nf, tpb), tpb>>>(R, D, nf);
    }
    double t6 = sync_now();

    // 6. results back to the host
    res.recs.resize(nf);
    res.derived.resize(nf);
    double t7, t8;
    if (opt.pinned_out) {
        FlowRec* hr = (FlowRec*)host_buf(1, (size_t)nf * sizeof(FlowRec));
        FlowDerived* hd = (FlowDerived*)host_buf(2, (size_t)nf * sizeof(FlowDerived));
        CK(cudaMemcpy(hr, R, (size_t)nf * sizeof(FlowRec), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hd, D, (size_t)nf * sizeof(FlowDerived), cudaMemcpyDeviceToHost));
        t7 = sync_now();
        // Copy into the result vectors with all host threads: the vectors are
        // freshly allocated, so this is mostly page-fault time.
        const long nblk = ((long)nf + 16383) / 16384;
        #pragma omp parallel for schedule(static)
        for (long b = 0; b < nblk; b++) {
            size_t lo = (size_t)b * 16384, cnt = (lo + 16384 <= nf) ? 16384 : nf - lo;
            memcpy(res.recs.data() + lo, hr + lo, cnt * sizeof(FlowRec));
            memcpy(res.derived.data() + lo, hd + lo, cnt * sizeof(FlowDerived));
        }
        t8 = now_s();
    } else {
        CK(cudaMemcpy(res.recs.data(), R, (size_t)nf * sizeof(FlowRec), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(res.derived.data(), D, (size_t)nf * sizeof(FlowDerived), cudaMemcpyDeviceToHost));
        t7 = sync_now();
        t8 = t7;
    }
    res.times.add("h2d", t1 - t0);
    res.times.add("flow_id", t2 - t1);
    res.times.add("sort", t3 - t2);
    res.times.add("gather", t4 - t3);
    res.times.add("reduce", t5 - t4);
    res.times.add("finalize", t6 - t5);
    res.times.add("d2h", t7 - t6);
    res.times.add("host_copy", t8 - t7);
    res.times.total = t8 - t0;
    return res;
}

}  // namespace ff
