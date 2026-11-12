// Shared definitions for every implementation (sequential, OpenMP, CUDA, MPI4Py).
//
// A packet is five 32-bit words. A flow is the set of packets with the same
// 5-tuple (source address, destination address, source port, destination port,
// protocol), in one direction. For each flow we keep exact integer accumulators
// (FlowRec) and derive the statistical features from them (FlowDerived) with
// the same code on the CPU and on the GPU, so results can be compared exactly.
#pragma once
#include <cstdint>
#include <cstddef>
#include <cmath>

#ifdef __CUDACC__
#define FF_HD __host__ __device__
#else
#define FF_HD
#endif

namespace ff {

// ---------------------------------------------------------------- packets
struct Packet {
    uint32_t ts;     // microseconds since the start of the batch (non-decreasing)
    uint32_t src;    // IPv4 source address
    uint32_t dst;    // IPv4 destination address
    uint32_t ports;  // source port << 16 | destination port
    uint32_t meta;   // protocol << 24 | TCP flags << 16 | IP length in bytes
};
static_assert(sizeof(Packet) == 20, "Packet must be 20 bytes");

FF_HD inline uint32_t pk_len(uint32_t meta) { return meta & 0xFFFFu; }
FF_HD inline uint32_t pk_flags(uint32_t meta) { return (meta >> 16) & 0xFFu; }
FF_HD inline uint32_t pk_proto(uint32_t meta) { return meta >> 24; }
inline uint32_t make_meta(uint32_t proto, uint32_t flags, uint32_t len) {
    return (proto << 24) | ((flags & 0xFFu) << 16) | (len & 0xFFFFu);
}

// A batch may span at most 2^31 microseconds (35.8 minutes). This keeps the
// sum of squared inter-arrival times below 2^62, so 64-bit sums cannot overflow.
static const uint32_t MAX_BATCH_US = 0x7FFFFFFFu;

// ---------------------------------------------------------------- features
static const int NFLAGS = 6;        // FIN, SYN, RST, PSH, ACK, URG (TCP flag bits 0..5)
static const int NBINS = 16;        // packet-length histogram: 100-byte bins, last bin open
static const uint32_t BIN_WIDTH = 100;
static const uint32_t DEFAULT_GAP_US = 1000;  // a gap >= 1 ms ends a burst

FF_HD inline uint32_t len_bin(uint32_t len) {
    uint32_t b = len / BIN_WIDTH;
    return b < (uint32_t)NBINS ? b : (uint32_t)(NBINS - 1);
}

// Exact per-flow accumulators. 176 bytes, no padding.
struct FlowRec {
    uint64_t bytes;      // sum of packet lengths
    uint64_t sum_len2;   // sum of squared packet lengths
    uint64_t sum_iat2;   // sum of squared inter-arrival times (microseconds^2)
    uint32_t src, dst, ports, proto;   // the flow key
    uint32_t n;                        // packets
    uint32_t len_min, len_max;
    uint32_t first_ts, last_ts;
    uint32_t iat_min, iat_max;         // inter-arrival time extremes (0 if n < 2)
    uint32_t bursts;                   // number of bursts
    uint32_t max_burst;                // packets in the longest burst
    uint32_t head_burst, tail_burst;   // working state for merging; 0 in the output
    uint32_t flags[NFLAGS];            // packets carrying each TCP flag
    uint32_t hist[NBINS];              // packet-length histogram
    uint32_t pad;
};
static_assert(sizeof(FlowRec) == 176, "FlowRec must be 176 bytes");

// Statistical features derived from a FlowRec.
struct FlowDerived {
    double duration_s;   // last - first timestamp
    double len_mean, len_std;
    double iat_mean, iat_std;      // microseconds
    double pkt_rate, byte_rate;    // per second (0 if duration is 0)
    double entropy;                // Shannon entropy of the length histogram, bits
    double mean_burst;             // packets per burst
};

// ---- 128-bit helpers so that variances are computed exactly on CPU and GPU
struct U128 { uint64_t hi, lo; };
FF_HD inline U128 mul64(uint64_t a, uint64_t b) {
    uint64_t a0 = (uint32_t)a, a1 = a >> 32, b0 = (uint32_t)b, b1 = b >> 32;
    uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    uint64_t mid = (p00 >> 32) + (uint32_t)p01 + (uint32_t)p10;
    U128 r;
    r.lo = (mid << 32) | (uint32_t)p00;
    r.hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return r;
}
// a - b for a >= b
FF_HD inline U128 sub128(U128 a, U128 b) {
    U128 r;
    r.lo = a.lo - b.lo;
    r.hi = a.hi - b.hi - (a.lo < b.lo ? 1u : 0u);
    return r;
}
FF_HD inline double to_double(U128 a) {
    return (double)a.hi * 18446744073709551616.0 + (double)a.lo;
}
// Population standard deviation of k samples with sum s and sum of squares q:
// sqrt(k*q - s*s) / k, with the numerator computed exactly in 128 bits.
FF_HD inline double exact_std(uint64_t k, uint64_t s, uint64_t q) {
    if (k == 0) return 0.0;
    U128 num = sub128(mul64(k, q), mul64(s, s));
    return sqrt(to_double(num)) / (double)k;
}

FF_HD inline FlowDerived derive(const FlowRec& r) {
    FlowDerived d;
    uint64_t dur = (uint64_t)(r.last_ts - r.first_ts);
    d.duration_s = (double)dur * 1e-6;
    d.len_mean = (double)r.bytes / (double)r.n;
    d.len_std = exact_std(r.n, r.bytes, r.sum_len2);
    if (r.n > 1) {
        d.iat_mean = (double)dur / (double)(r.n - 1);
        d.iat_std = exact_std(r.n - 1, dur, r.sum_iat2);
    } else {
        d.iat_mean = 0.0;
        d.iat_std = 0.0;
    }
    if (dur > 0) {
        d.pkt_rate = (double)r.n / d.duration_s;
        d.byte_rate = (double)r.bytes / d.duration_s;
    } else {
        d.pkt_rate = 0.0;
        d.byte_rate = 0.0;
    }
    double h = 0.0;
    for (int b = 0; b < NBINS; b++) {
        if (r.hist[b]) {
            double p = (double)r.hist[b] / (double)r.n;
            h -= p * log2(p);
        }
    }
    d.entropy = h;
    d.mean_burst = (double)r.n / (double)r.bursts;
    return d;
}

// ---------------------------------------------------------------- hashing
FF_HD inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}
FF_HD inline uint64_t flow_hash(uint32_t src, uint32_t dst, uint32_t ports, uint32_t proto) {
    uint64_t a = ((uint64_t)src << 32) | dst;
    uint64_t b = ((uint64_t)ports << 8) | proto;
    return mix64(a ^ mix64(b + 0x9e3779b97f4a7c15ULL));
}
FF_HD inline uint64_t pk_hash(const Packet& p) {
    return flow_hash(p.src, p.dst, p.ports, pk_proto(p.meta));
}
FF_HD inline bool same_flow(const Packet& a, const Packet& b) {
    return a.src == b.src && a.dst == b.dst && a.ports == b.ports &&
           ((a.meta ^ b.meta) >> 24) == 0;
}

// ---------------------------------------------------------------- updates
// Start a record from its first packet.
FF_HD inline void rec_init(FlowRec& r, const Packet& p) {
    uint32_t len = pk_len(p.meta), fl = pk_flags(p.meta);
    r.bytes = len;
    r.sum_len2 = (uint64_t)len * len;
    r.sum_iat2 = 0;
    r.src = p.src; r.dst = p.dst; r.ports = p.ports; r.proto = pk_proto(p.meta);
    r.n = 1;
    r.len_min = r.len_max = len;
    r.first_ts = r.last_ts = p.ts;
    r.iat_min = 0xFFFFFFFFu; r.iat_max = 0;
    r.bursts = 1; r.max_burst = 1; r.head_burst = 0; r.tail_burst = 1;
    for (int k = 0; k < NFLAGS; k++) r.flags[k] = (fl >> k) & 1u;
    for (int b = 0; b < NBINS; b++) r.hist[b] = 0;
    r.hist[len_bin(len)] = 1;
    r.pad = 0;
}

// Add the next packet (in time order) of the flow.
FF_HD inline void rec_update(FlowRec& r, const Packet& p, uint32_t gap) {
    uint32_t len = pk_len(p.meta), fl = pk_flags(p.meta);
    uint32_t iat = p.ts - r.last_ts;
    r.last_ts = p.ts;
    if (iat < r.iat_min) r.iat_min = iat;
    if (iat > r.iat_max) r.iat_max = iat;
    r.sum_iat2 += (uint64_t)iat * iat;
    if (iat < gap) {
        r.tail_burst++;
        if (r.tail_burst > r.max_burst) r.max_burst = r.tail_burst;
    } else {
        if (r.bursts == 1) r.head_burst = r.tail_burst;  // length of the first burst
        r.bursts++;
        r.tail_burst = 1;
    }
    r.n++;
    r.bytes += len;
    r.sum_len2 += (uint64_t)len * len;
    if (len < r.len_min) r.len_min = len;
    if (len > r.len_max) r.len_max = len;
    for (int k = 0; k < NFLAGS; k++) r.flags[k] += (fl >> k) & 1u;
    r.hist[len_bin(len)]++;
}

// Merge record b (later in time) into a (earlier). Both describe the same flow
// over two consecutive time ranges. The operation is associative, so partial
// records computed by different threads can be combined in time order.
inline void rec_merge(FlowRec& a, const FlowRec& b, uint32_t gap) {
    uint32_t a_head = a.bursts == 1 ? a.n : a.head_burst;
    uint32_t b_head = b.bursts == 1 ? b.n : b.head_burst;
    uint32_t iat = b.first_ts - a.last_ts;
    uint32_t imin = a.iat_min < b.iat_min ? a.iat_min : b.iat_min;
    uint32_t imax = a.iat_max > b.iat_max ? a.iat_max : b.iat_max;
    a.iat_min = iat < imin ? iat : imin;
    a.iat_max = iat > imax ? iat : imax;
    a.sum_iat2 += b.sum_iat2 + (uint64_t)iat * iat;
    uint32_t mb = a.max_burst > b.max_burst ? a.max_burst : b.max_burst;
    if (iat < gap) {  // the last burst of a and the first burst of b are one burst
        uint32_t joined = a.tail_burst + b_head;
        uint32_t nh = a.bursts == 1 ? joined : a_head;
        uint32_t nt = b.bursts == 1 ? joined : b.tail_burst;
        a.max_burst = joined > mb ? joined : mb;
        a.bursts = a.bursts + b.bursts - 1;
        a.head_burst = nh; a.tail_burst = nt;
    } else {
        a.max_burst = mb;
        a.bursts += b.bursts;
        a.head_burst = a_head; a.tail_burst = b.tail_burst;
    }
    a.n += b.n;
    a.bytes += b.bytes;
    a.sum_len2 += b.sum_len2;
    if (b.len_min < a.len_min) a.len_min = b.len_min;
    if (b.len_max > a.len_max) a.len_max = b.len_max;
    a.last_ts = b.last_ts;
    for (int k = 0; k < NFLAGS; k++) a.flags[k] += b.flags[k];
    for (int k = 0; k < NBINS; k++) a.hist[k] += b.hist[k];
}

// Put a record into its output form.
FF_HD inline void rec_finish(FlowRec& r) {
    r.head_burst = 0; r.tail_burst = 0; r.pad = 0;
    if (r.n < 2) r.iat_min = 0;
}

}  // namespace ff
