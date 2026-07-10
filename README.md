# OpenCL Custom Tensor Core (C++ & OpenCL)

A high-performance GPGPU matrix multiplication and activation engine written from scratch in C++ and OpenCL 1.2. The project implements naive and local-memory tiled compute kernels to accelerate deep learning operations directly on legacy AMD Radeon GPUs (tested on AMD Radeon HD 7500 Series, "Turks" core, 1GB VRAM) and benchmarks them against a single-threaded CPU baseline.

---

## Features

- **Custom OpenCL Kernels**:
  - **Naive MatMul**: Baseline GPU matrix multiplication.
  - **Local Memory Tiled MatMul**: Cache-efficient implementation utilizing GPU shared SRAM (`__local` memory) to minimize VRAM bandwidth bottlenecks (cooperative tiles of size $16 \times 16$).
  - **Standalone Activation**: Out-of-place activation pass supporting **ReLU** and **Sigmoid**.
  - **Fused Kernels**: Combined matrix multiplication and activation in a single GPU pass to eliminate intermediate memory round-trips to VRAM.
- **Robust Host Driver (C++)**:
  - Automatically query and select AMD Accelerated Parallel Processing GPU platforms.
  - Compiles OpenCL compute shaders (`.cl` files) at runtime.
  - Performance profiling utilizing high-resolution host clocks and GPU hardware profiling events.
  - Outputs detailed metrics: PCIe bandwidth (GB/s), compute throughput (GFLOPS), speedup factors, and error tolerances.
- **TDR Safety System**:
  - Automatically detects large workloads ($\ge 1024^3$ FLOPs) and skips the slow naive GPU implementation, preventing display driver timeouts (Windows TDR) while executing the fast tiled engine safely.

---

## Technical Insights & Architecture

### Tiled Memory Cache Optimization
Matrix multiplication $C = A \times B$ is memory-bandwidth intensive. In the naive version, every thread reads rows of $A$ and columns of $B$ repeatedly from global VRAM, bottlenecking the GPU.
Our optimized tiled version divides the grid into $16 \times 16$ thread blocks. Threads cooperatively cache chunks of matrices $A$ and $B$ into fast, on-chip shared `__local` SRAM memory. By synchronizing threads using `barrier(CLK_LOCAL_MEM_FENCE)`, the engine loads memory once and reuses it 16 times, boosting compute throughput by **5.3x** on the Turks GPU architecture.

### CPU Cache Thrashing
For small matrix sizes, the CPU performs at $\sim 1.03$ GFLOPS. However, at size 1536, the CPU throughput plummets to **0.17 GFLOPS**. This is due to cache line eviction; a $1536 \times 1536$ single-precision matrix requires 9 MB of memory, exceeding the L1/L2 caches. Reading B in column-major format forces frequent cache misses, whereas the GPU's tiling and thread scheduler hide memory latency.

---

## System Requirements

- **Operating System**: Windows.
- **GPU Driver**: AMD Catalyst / Radeon driver with OpenCL support.
- **OpenCL Runtime**: `OpenCL.dll` (located in `C:\Windows\System32\OpenCL.dll`).
- **Compiler**: GCC / G++ (MSYS2 UCRT64 toolchain is recommended).

---

## Getting Started

### 0. Fetch Dependencies
Since OpenCL headers are third-party dependencies, they are ignored by Git. Before compiling, fetch them from the official Khronos Group repository:
```bash
git clone --depth 1 https://github.com/KhronosGroup/OpenCL-Headers.git third_party/OpenCL-Headers
```

### 1. Compilation
Build the application utilizing the provided `Makefile`. Open your terminal and run:
```bash
# Compile using MSYS2 Make
mingw32-make

# Or compile manually with g++
g++ -O3 -Wall -Ithird_party/OpenCL-Headers src/main.cpp -o opencl_matmul.exe C:\Windows\System32\OpenCL.dll
g++ -O3 -Wall -Ithird_party/OpenCL-Headers src/query_device.cpp -o query_device.exe C:\Windows\System32\OpenCL.dll
```

### 2. Query GPU Device Info
Run the query binary to inspect your OpenCL platform name, driver version, and workgroup size limits:
```bash
./query_device.exe
```

### 3. Running Benchmarks
Execute the main application with customizable CLI options:
```bash
# Run a quick square matrix test with validation and ReLU activation
./opencl_matmul.exe --size 512 --activation relu

# Force run validation on a larger size
./opencl_matmul.exe --size 1024 --activation sigmoid --validate

# Run a large benchmark with safe TDR auto-skipping on naive kernels
./opencl_matmul.exe --size 1536 --activation relu

# Run custom dimension rectangular matrices A(512x256) * B(256x1024)
./opencl_matmul.exe -m 512 -k 256 -n 1024
```

### CLI Command Options
* `-s, --size <size>`: Set square matrix dimension (default: `512`).
* `-m <rows>, -n <cols>, -k <inner>`: Set custom dimensions.
* `-a, --activation <type>`: Select activation function: `none`, `relu`, `sigmoid` (default: `none`).
* `-v, --validate`: Force check GPU output correctness against CPU.
* `--no-validate`: Disable CPU validation.
* `--skip-naive`: Manually force skip the naive GPU benchmarks.
* `-r, --runs <count>`: Number of iterations to run GPU benchmarks to compute average timing (default: `5`).

---

## Benchmark Highlights (AMD Radeon Turks Core)

For a **1536x1536** matrix multiplication:
- **CPU (Single-Thread)**: 43.6 seconds (0.17 GFLOPS)
- **GPU Tiled (Fused ReLU)**: **1.11 seconds** (6.50 GFLOPS)
- **Performance Increase**: **39.11x Speedup** (Kernel only) or **20.47x Speedup** (overall, including PCIe transfers)
- **Numerical Parity**: Verified with maximum absolute error of `0.00e+00`.
- **PCIe H2D Upload**: **0.98 GB/s**
- **PCIe D2H Download**: **2.88 GB/s**
