# Builds the flow feature extraction programs.
#
#   make            CPU-only build (sequential + OpenMP)
#   make CUDA=1     adds the CUDA implementations (needs nvcc)
#
# macOS: Apple clang has no OpenMP, so Homebrew GCC is used (brew install gcc).
# CUDA: ARCH=sm_61 targets the Quadro P2000 (Pascal); e.g. ARCH=sm_75 for a T4.
UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
CXX      = g++-16
NATIVE   = -mcpu=native
else
CXX     ?= g++
NATIVE   = -march=native
endif
CXXFLAGS = -O3 $(NATIVE) -std=c++17 -fopenmp -Wall -Wextra -fPIC
NVCC    ?= nvcc
ARCH    ?= sm_61
HOSTCXX ?= g++-12
NVCCFLAGS = -O3 -std=c++17 -arch=$(ARCH) -ccbin $(HOSTCXX) -Xcompiler -fPIC -Xcompiler -fopenmp
CUDALIBS ?= -lcudart

B = build
CORE = $(B)/cpu.o $(B)/util.o
ifeq ($(CUDA),1)
GPUOBJ = $(B)/gpu.o
LIBS = $(CUDALIBS)
else
GPUOBJ = $(B)/gpu_stub.o
LIBS =
endif

all: flowfeat pcap2pkt contention_bench libflowfeat.so

$(B):
	mkdir -p $(B)
$(B)/%.o: src/%.cpp src/common.h src/flowfeat.h | $(B)
	$(CXX) $(CXXFLAGS) -c -o $@ $<
$(B)/gpu.o: src/gpu.cu src/common.h src/flowfeat.h | $(B)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

flowfeat: $(B)/main.o $(CORE) $(GPUOBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)
pcap2pkt: $(B)/pcap2pkt.o $(CORE)
	$(CXX) $(CXXFLAGS) -o $@ $^
contention_bench: $(B)/contention_bench.o $(CORE)
	$(CXX) $(CXXFLAGS) -o $@ $^
libflowfeat.so: $(B)/capi.o $(CORE) $(GPUOBJ)
	$(CXX) $(CXXFLAGS) -shared -o $@ $^ $(LIBS)

clean:
	rm -rf $(B) flowfeat pcap2pkt contention_bench libflowfeat.so
.PHONY: all clean
