// Stand-in for gpu.cu on machines without CUDA.
#include "flowfeat.h"
#include <cstdio>
#include <cstdlib>
namespace ff {
bool gpu_available() { return false; }
std::string gpu_name() { return "none"; }
Result run_gpu(const Packet*, size_t, uint32_t, const GpuOpts&) {
    fprintf(stderr, "This binary was built without CUDA (make CUDA=1).\n");
    exit(2);
}
}  // namespace ff
