// pcap2pkt: convert a packet capture (classic pcap format) into .pkt batches.
//
//   pcap2pkt capture.pcap out_prefix [--window SECONDS] [--max-packets N] [--max-seconds S]
//   gunzip -c capture.pcap.gz | pcap2pkt - out_prefix --window 60
//
// Only IPv4 packets are kept. The flow key uses the ports for TCP and UDP and
// zero ports for every other protocol. The length is the IP total length, so
// captures truncated to a snapshot length keep the original packet sizes.
// With --window, one file per time window is written: out_prefix_0000.pkt, ...
// Timestamps inside a file are microseconds since the start of its window.
#include "flowfeat.h"
#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace ff;

static inline uint16_t be16(const uint8_t* p) { return (uint16_t)((p[0] << 8) | p[1]); }
static inline uint32_t be32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
}
static inline uint32_t swap32(uint32_t x) {
    return (x >> 24) | ((x >> 8) & 0xFF00u) | ((x << 8) & 0xFF0000u) | (x << 24);
}

// Parse one captured frame. Returns false if it is not an IPv4 packet.
static bool parse(const uint8_t* d, uint32_t caplen, uint32_t origlen, uint32_t linktype, Packet* out) {
    uint32_t off = 0;
    if (linktype == 1) {  // Ethernet, with optional VLAN tags
        if (caplen < 14) return false;
        uint16_t et = be16(d + 12);
        off = 14;
        while ((et == 0x8100 || et == 0x88a8) && caplen >= off + 4) { et = be16(d + off + 2); off += 4; }
        if (et != 0x0800) return false;
    } else if (linktype == 113) {  // Linux cooked capture
        if (caplen < 16 || be16(d + 14) != 0x0800) return false;
        off = 16;
    } else if (linktype == 101 || linktype == 12 || linktype == 228) {  // raw IP
        off = 0;
    } else {
        return false;
    }
    if (caplen < off + 20) return false;
    const uint8_t* ip = d + off;
    if ((ip[0] >> 4) != 4) return false;
    uint32_t ihl = (ip[0] & 15u) * 4;
    if (ihl < 20) return false;
    uint32_t len = be16(ip + 2);
    if (len == 0) len = origlen > off ? origlen - off : 0;  // segmentation offload
    if (len > 0xFFFF) len = 0xFFFF;
    uint32_t proto = ip[9], ports = 0, flags = 0;
    bool first_frag = (be16(ip + 6) & 0x1FFF) == 0;
    const uint8_t* l4 = ip + ihl;
    uint32_t have = caplen - off;
    if (first_frag && (proto == 6 || proto == 17) && have >= ihl + 4) {
        ports = ((uint32_t)be16(l4) << 16) | be16(l4 + 2);
        if (proto == 6 && have >= ihl + 14) flags = l4[13] & 0x3Fu;
    }
    out->src = be32(ip + 12);
    out->dst = be32(ip + 16);
    out->ports = ports;
    out->meta = make_meta(proto, flags, len);
    return true;
}

struct Writer {
    std::string prefix;
    bool windowed;
    int index = 0;
    size_t files = 0, packets = 0, resorted = 0;
    bool flush(PacketVec& pk, uint64_t base) {
        if (pk.empty()) return true;
        // Captures are almost always in time order. If not, restore it (stable).
        bool sorted = true;
        for (size_t i = 1; i < pk.size(); i++) if (pk[i].ts < pk[i - 1].ts) { sorted = false; break; }
        if (!sorted) {
            std::stable_sort(pk.begin(), pk.end(), [](const Packet& a, const Packet& b) { return a.ts < b.ts; });
            resorted++;
        }
        char name[1024];
        if (windowed) snprintf(name, sizeof name, "%s_%04d.pkt", prefix.c_str(), index);
        else snprintf(name, sizeof name, "%s.pkt", prefix.c_str());
        if (!write_pkt(name, pk.data(), pk.size(), base)) { fprintf(stderr, "cannot write %s\n", name); return false; }
        printf("%s: %zu packets, %.3f s\n", name, pk.size(), (pk.back().ts - pk.front().ts) * 1e-6);
        files++; packets += pk.size();
        pk.clear();
        return true;
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: pcap2pkt <capture.pcap | -> <out_prefix> [--window S] [--max-packets N] [--max-seconds S]\n");
        return 2;
    }
    std::string in = argv[1];
    Writer w;
    w.prefix = argv[2];
    double window_s = 0.0, max_s = 0.0;
    uint64_t max_packets = 0;
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--window" && i + 1 < argc) window_s = atof(argv[++i]);
        else if (a == "--max-packets" && i + 1 < argc) max_packets = strtoull(argv[++i], nullptr, 10);
        else if (a == "--max-seconds" && i + 1 < argc) max_s = atof(argv[++i]);
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    w.windowed = window_s > 0.0;
    uint64_t window_us = w.windowed ? (uint64_t)(window_s * 1e6) : (uint64_t)MAX_BATCH_US;
    if (window_us > MAX_BATCH_US) { fprintf(stderr, "window too long (max %.0f s)\n", MAX_BATCH_US * 1e-6); return 2; }

    FILE* f = in == "-" ? stdin : fopen(in.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", in.c_str()); return 1; }
    static char iobuf[1 << 22];
    setvbuf(f, iobuf, _IOFBF, sizeof iobuf);
    uint32_t gh[6];
    if (fread(gh, 4, 6, f) != 6) { fprintf(stderr, "not a pcap file\n"); return 1; }
    bool swapped, nanos;
    switch (gh[0]) {
        case 0xa1b2c3d4u: swapped = false; nanos = false; break;
        case 0xa1b23c4du: swapped = false; nanos = true; break;
        case 0xd4c3b2a1u: swapped = true; nanos = false; break;
        case 0x4d3cb2a1u: swapped = true; nanos = true; break;
        default: fprintf(stderr, "not a classic pcap file (pcapng is not supported; convert with editcap -F pcap)\n"); return 1;
    }
    uint32_t linktype = swapped ? swap32(gh[5]) : gh[5];

    PacketVec pk;
    std::vector<uint8_t> data(1 << 18);
    uint64_t total = 0, kept = 0, skipped = 0, t_first = 0, base = 0;
    bool have_first = false;
    for (;;) {
        uint32_t ph[4];
        if (fread(ph, 4, 4, f) != 4) break;
        if (swapped) for (int k = 0; k < 4; k++) ph[k] = swap32(ph[k]);
        uint32_t caplen = ph[2];
        if (caplen > data.size()) {
            if (caplen > (1u << 26)) { fprintf(stderr, "corrupt record\n"); break; }
            data.resize(caplen);
        }
        if (fread(data.data(), 1, caplen, f) != caplen) break;
        total++;
        uint64_t t = (uint64_t)ph[0] * 1000000ULL + (nanos ? ph[1] / 1000 : ph[1]);
        if (!have_first) { have_first = true; t_first = t; base = t; }
        if (max_s > 0.0 && t >= t_first + (uint64_t)(max_s * 1e6)) break;
        Packet p;
        if (!parse(data.data(), caplen, ph[3], linktype, &p)) { skipped++; continue; }
        if (t < base) t = base;  // slightly out-of-order record at a window boundary
        if (t - base >= window_us) {
            if (!w.flush(pk, base)) return 1;
            if (!w.windowed) break;  // a single batch holds at most MAX_BATCH_US
            uint64_t k = (t - t_first) / window_us;
            base = t_first + k * window_us;
            w.index = (int)k;
        }
        p.ts = (uint32_t)(t - base);
        pk.push_back(p);
        kept++;
        if (max_packets && kept >= max_packets) break;
    }
    if (!w.flush(pk, base)) return 1;
    printf("records read %" PRIu64 ", IPv4 packets kept %" PRIu64 ", other records skipped %" PRIu64
           ", files %zu, files re-sorted by time %zu\n", total, kept, skipped, w.files, w.resorted);
    return 0;
}
