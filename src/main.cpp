#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <CL/cl.h>

#define TS 16  // Tiling size (TS x TS = 256 threads, within GPU max 256)

// Structure to store benchmark results
struct BenchmarkResult {
    double kernel_time_ms = 0.0;
    double total_time_ms = 0.0;  // includes H2D and D2H transfers
    double gflops = 0.0;
    double bandwidth_gb_s = 0.0;
    bool success = false;
};

// OpenCL Error helper
const char* getOpenCLErrorString(cl_int error) {
    switch(error){
        // run-time and sgim errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
        // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_GL_OBJECT";
        case -60: return "CL_INVALID_BUFFER_SIZE";
        case -61: return "CL_INVALID_MIP_LEVEL";
        case -62: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -63: return "CL_INVALID_PROPERTY";
        default: return "Unknown OpenCL error";
    }
}

void checkError(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        std::cerr << "Error during " << operation << ": " << getOpenCLErrorString(err) << " (code: " << err << ")" << std::endl;
        exit(1);
    }
}

// Read kernel file
std::string readKernelFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open kernel file: " << filepath << std::endl;
        exit(1);
    }
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

// Single-threaded CPU MatMul baseline
void cpu_matmul(int M, int N, int K, const float* A, const float* B, float* C) {
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[r * K + k] * B[k * N + c];
            }
            C[r * N + c] = sum;
        }
    }
}

// CPU Activation function
void cpu_activation(int M, int N, float* C, int activation_type) {
    int size = M * N;
    if (activation_type == 1) { // ReLU
        for (int i = 0; i < size; ++i) {
            C[i] = C[i] > 0.0f ? C[i] : 0.0f;
        }
    } else if (activation_type == 2) { // Sigmoid
        for (int i = 0; i < size; ++i) {
            C[i] = 1.0f / (1.0f + std::exp(-C[i]));
        }
    }
}

int main(int argc, char* argv[]) {
    // Defaults
    int size = 512;
    int M = -1, N = -1, K = -1;
    std::string act_str = "none";
    int activation_type = 0; // 0: None, 1: ReLU, 2: Sigmoid
    bool validate = true;
    bool validate_set = false;
    int runs = 5;
    bool skip_naive = false;
    bool skip_naive_set = false;

    // Command-line parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-s" || arg == "--size") {
            size = std::stoi(argv[++i]);
        } else if (arg == "-m") {
            M = std::stoi(argv[++i]);
        } else if (arg == "-n") {
            N = std::stoi(argv[++i]);
        } else if (arg == "-k") {
            K = std::stoi(argv[++i]);
        } else if (arg == "-a" || arg == "--activation") {
            act_str = argv[++i];
            std::transform(act_str.begin(), act_str.end(), act_str.begin(), ::tolower);
        } else if (arg == "-v" || arg == "--validate") {
            validate = true;
            validate_set = true;
        } else if (arg == "--no-validate") {
            validate = false;
            validate_set = true;
        } else if (arg == "--skip-naive") {
            skip_naive = true;
            skip_naive_set = true;
        } else if (arg == "--no-skip-naive") {
            skip_naive = false;
            skip_naive_set = true;
        } else if (arg == "-r" || arg == "--runs") {
            runs = std::stoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  -s, --size <size>          Set square matrix size (default: 512)\n"
                      << "  -m <rows>, -n <cols>, -k <inner> Set custom dimensions\n"
                      << "  -a, --activation <type>    Activation: none, relu, sigmoid (default: none)\n"
                      << "  -v, --validate             Force CPU validation\n"
                      << "  --no-validate              Disable CPU validation\n"
                      << "  --skip-naive               Skip naive GPU kernels to avoid GPU timeouts (default: auto for large sizes)\n"
                      << "  -r, --runs <count>         Number of iterations for GPU benchmark (default: 5)\n"
                      << "  -h, --help                 Display this help\n";
            return 0;
        }
    }

    if (M == -1) M = size;
    if (N == -1) N = size;
    if (K == -1) K = size;

    if (act_str == "relu") activation_type = 1;
    else if (act_str == "sigmoid") activation_type = 2;

    // Automatic validation override for large matrices to save time
    if (!validate_set) {
        if (M * N * K > 512 * 512 * 512) {
            validate = false;
        }
    }

    if (!skip_naive_set) {
        // Automatically skip naive GPU kernels for size >= 1024 to prevent Windows TDR (driver timeout)
        if (static_cast<long long>(M) * N * K >= 1024LL * 1024 * 1024) {
            skip_naive = true;
        }
    }

    std::cout << "========================================================\n";
    std::cout << " OpenCL Custom Tensor Core Benchmark\n";
    std::cout << "========================================================\n";
    std::cout << "Matrix Dimensions: A(" << M << "x" << K << ") * B(" << K << "x" << N << ") = C(" << M << "x" << N << ")\n";
    std::cout << "Matrix Sizes: A: " << (M*K*sizeof(float)/1024.0/1024.0) << " MB, B: " 
              << (K*N*sizeof(float)/1024.0/1024.0) << " MB, C: " << (M*N*sizeof(float)/1024.0/1024.0) << " MB\n";
    std::cout << "Total Memory Required: " << ((M*K + K*N + M*N)*sizeof(float)/1024.0/1024.0) << " MB\n";
    std::cout << "Activation Function: " << act_str << "\n";
    std::cout << "Benchmark Runs:      " << runs << "\n";
    std::cout << "CPU Validation:      " << (validate ? "Enabled" : "Disabled") << "\n";
    std::cout << "Skip Naive GPU:      " << (skip_naive ? "Yes (Auto/Forced)" : "No") << "\n";
    std::cout << "========================================================\n";

    // ------------------------------------------------------------------------
    // Host Allocation and Initialization
    // ------------------------------------------------------------------------
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_cpu(M * N, 0.0f);
    std::vector<float> h_C_gpu_naive(M * N, 0.0f);
    std::vector<float> h_C_gpu_tiled(M * N, 0.0f);
    std::vector<float> h_C_gpu_naive_fused(M * N, 0.0f);
    std::vector<float> h_C_gpu_tiled_fused(M * N, 0.0f);

    // Initialize matrices A and B with pseudo-random floats in [-1.0, 1.0]
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
    }

    // ------------------------------------------------------------------------
    // CPU Execution
    // ------------------------------------------------------------------------
    double cpu_time_ms = 0.0;
    if (validate) {
        std::cout << "\n[CPU] Running baseline matrix multiplication..." << std::endl;
        auto t_start = std::chrono::high_resolution_clock::now();
        cpu_matmul(M, N, K, h_A.data(), h_B.data(), h_C_cpu.data());
        cpu_activation(M, N, h_C_cpu.data(), activation_type);
        auto t_end = std::chrono::high_resolution_clock::now();
        cpu_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        double cpu_gflops = (2.0 * M * N * K) / (cpu_time_ms / 1000.0) / 1e9;
        std::cout << "[CPU] Finished in " << std::fixed << std::setprecision(2) << cpu_time_ms << " ms (" << cpu_gflops << " GFLOPS)" << std::endl;
    } else {
        std::cout << "\n[CPU] Skipped." << std::endl;
    }

    // ------------------------------------------------------------------------
    // OpenCL Setup
    // ------------------------------------------------------------------------
    std::cout << "\n[GPU] Initializing OpenCL..." << std::endl;

    cl_uint numPlatforms = 0;
    checkError(clGetPlatformIDs(0, nullptr, &numPlatforms), "clGetPlatformIDs (count)");
    std::vector<cl_platform_id> platforms(numPlatforms);
    checkError(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr), "clGetPlatformIDs");

    // Select the first platform (AMD APP)
    cl_platform_id platform = platforms[0];
    char pName[256];
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(pName), pName, nullptr);
    std::cout << "[GPU] Platform: " << pName << std::endl;

    // Find GPU devices
    cl_uint numDevices = 0;
    cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if (err != CL_SUCCESS || numDevices == 0) {
        std::cerr << "Error: No GPU devices found on platform." << std::endl;
        return 1;
    }
    std::vector<cl_device_id> devices(numDevices);
    checkError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr), "clGetDeviceIDs");

    cl_device_id device = devices[0];
    char dName[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dName), dName, nullptr);
    std::cout << "[GPU] Device:   " << dName << std::endl;

    // Create Context and Command Queue
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    checkError(err, "clCreateContext");

    // Use clCreateCommandQueue which is compatible with OpenCL 1.2
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    checkError(err, "clCreateCommandQueue");

    // Load and build program
    std::string kernelSource = readKernelFile("kernels/matmul.cl");
    const char* sourcePtr = kernelSource.c_str();
    size_t sourceLen = kernelSource.length();

    cl_program program = clCreateProgramWithSource(context, 1, &sourcePtr, &sourceLen, &err);
    checkError(err, "clCreateProgramWithSource");

    // Compile with TS definition
    std::string buildOpts = "-DTS=" + std::to_string(TS);
    err = clBuildProgram(program, 1, &device, buildOpts.c_str(), nullptr, nullptr);
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> buildLog(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);
        std::cerr << "--- Kernel Build Log ---\n" << buildLog.data() << "\n-----------------------" << std::endl;
        exit(1);
    }
    checkError(err, "clBuildProgram");

    // Create kernels
    cl_kernel k_naive = clCreateKernel(program, "matmul_naive", &err);
    checkError(err, "clCreateKernel(matmul_naive)");
    cl_kernel k_tiled = clCreateKernel(program, "matmul_tiled", &err);
    checkError(err, "clCreateKernel(matmul_tiled)");
    cl_kernel k_act = clCreateKernel(program, "apply_activation", &err);
    checkError(err, "clCreateKernel(apply_activation)");
    cl_kernel k_naive_fused = clCreateKernel(program, "matmul_naive_fused", &err);
    checkError(err, "clCreateKernel(matmul_naive_fused)");
    cl_kernel k_tiled_fused = clCreateKernel(program, "matmul_tiled_fused", &err);
    checkError(err, "clCreateKernel(matmul_tiled_fused)");

    // ------------------------------------------------------------------------
    // Memory Transfers H2D (Host to Device)
    // ------------------------------------------------------------------------
    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY, bytesA, nullptr, &err);
    checkError(err, "clCreateBuffer A");
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY, bytesB, nullptr, &err);
    checkError(err, "clCreateBuffer B");
    cl_mem d_C = clCreateBuffer(context, CL_MEM_READ_WRITE, bytesC, nullptr, &err);
    checkError(err, "clCreateBuffer C");

    std::cout << "[GPU] Uploading matrices A and B (Host -> Device)..." << std::endl;
    
    // Benchmark H2D Bandwidth
    auto t_h2d_start = std::chrono::high_resolution_clock::now();
    checkError(clEnqueueWriteBuffer(queue, d_A, CL_TRUE, 0, bytesA, h_A.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer A");
    checkError(clEnqueueWriteBuffer(queue, d_B, CL_TRUE, 0, bytesB, h_B.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer B");
    clFinish(queue);
    auto t_h2d_end = std::chrono::high_resolution_clock::now();
    double h2d_time_ms = std::chrono::duration<double, std::milli>(t_h2d_end - t_h2d_start).count();
    double h2d_bandwidth = (double)(bytesA + bytesB) / 1024.0 / 1024.0 / 1024.0 / (h2d_time_ms / 1000.0);
    std::cout << "[GPU] H2D Upload Time:      " << h2d_time_ms << " ms (" << h2d_bandwidth << " GB/s)" << std::endl;

    // ------------------------------------------------------------------------
    // Benchmark Helpers
    // ------------------------------------------------------------------------
    // Computes flops and benchmarks specific kernel launch configurations
    auto run_gpu_benchmark = [&](cl_kernel kernel, bool fused, bool tiled, const char* name, std::vector<float>& h_out) -> BenchmarkResult {
        BenchmarkResult res;
        double kernel_time_sum = 0.0;
        double total_time_sum = 0.0;

        // Warmup run
        clSetKernelArg(kernel, 0, sizeof(int), &M);
        clSetKernelArg(kernel, 1, sizeof(int), &N);
        clSetKernelArg(kernel, 2, sizeof(int), &K);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_A);
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_B);
        clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_C);
        if (fused) {
            clSetKernelArg(kernel, 6, sizeof(int), &activation_type);
        }

        size_t global_size[2] = {
            ((size_t)M + TS - 1) / TS * TS,
            ((size_t)N + TS - 1) / TS * TS
        };
        size_t local_size[2] = { TS, TS };
        
        // Naive doesn't strictly need workgroup sizing but tiled does. We use TS x TS for both for matching conditions.
        cl_event event;
        err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_size, tiled ? local_size : nullptr, 0, nullptr, &event);
        checkError(err, "NDRangeKernel warmup");
        clFinish(queue);
        clReleaseEvent(event);

        // Benchmark runs
        for (int r = 0; r < runs; ++r) {
            // If separate activation, we also need to clear d_C and time it together
            auto t_total_start = std::chrono::high_resolution_clock::now();
            
            // Upload A and B (to measure total execution with transfers)
            clEnqueueWriteBuffer(queue, d_A, CL_FALSE, 0, bytesA, h_A.data(), 0, nullptr, nullptr);
            clEnqueueWriteBuffer(queue, d_B, CL_FALSE, 0, bytesB, h_B.data(), 0, nullptr, nullptr);
            
            cl_event main_event, act_event;
            cl_ulong start_time = 0, end_time = 0;

            // Execute Matrix Multiplication Kernel
            err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_size, tiled ? local_size : nullptr, 0, nullptr, &main_event);
            checkError(err, "NDRangeKernel main");

            double current_kernel_ms = 0.0;
            if (!fused && activation_type != 0) {
                // Separate activation run
                clSetKernelArg(k_act, 0, sizeof(int), &M);
                clSetKernelArg(k_act, 1, sizeof(int), &N);
                clSetKernelArg(k_act, 2, sizeof(cl_mem), &d_C);
                clSetKernelArg(k_act, 3, sizeof(int), &activation_type);
                
                err = clEnqueueNDRangeKernel(queue, k_act, 2, nullptr, global_size, nullptr, 0, nullptr, &act_event);
                checkError(err, "NDRangeKernel separate activation");
                
                clFinish(queue);
                
                // Get execution profile details
                clGetEventProfilingInfo(main_event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, nullptr);
                clGetEventProfilingInfo(main_event, CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, nullptr);
                current_kernel_ms += (double)(end_time - start_time) / 1e6;

                clGetEventProfilingInfo(act_event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, nullptr);
                clGetEventProfilingInfo(act_event, CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, nullptr);
                current_kernel_ms += (double)(end_time - start_time) / 1e6;

                clReleaseEvent(act_event);
            } else {
                // Fused or no activation
                clFinish(queue);
                clGetEventProfilingInfo(main_event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, nullptr);
                clGetEventProfilingInfo(main_event, CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, nullptr);
                current_kernel_ms = (double)(end_time - start_time) / 1e6;
            }
            clReleaseEvent(main_event);

            // Read output C back to host
            checkError(clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, bytesC, h_out.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer");
            clFinish(queue);

            auto t_total_end = std::chrono::high_resolution_clock::now();

            kernel_time_sum += current_kernel_ms;
            total_time_sum += std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();
        }

        res.kernel_time_ms = kernel_time_sum / runs;
        res.total_time_ms = total_time_sum / runs;
        res.gflops = (2.0 * M * N * K) / (res.kernel_time_ms / 1000.0) / 1e9;
        
        // Calculate effective kernel bandwidth
        // Read A: M * K * float
        // Read B: K * N * float
        // Write C: M * N * float (and if separate, we read and write C again)
        double data_moved_bytes = bytesA + bytesB + bytesC;
        if (!fused && activation_type != 0) {
            data_moved_bytes += 2 * bytesC; // separate read and write C
        }
        res.bandwidth_gb_s = data_moved_bytes / 1024.0 / 1024.0 / 1024.0 / (res.kernel_time_ms / 1000.0);

        // Verification correctness
        if (validate) {
            float max_err = 0.0f;
            for (int i = 0; i < M * N; ++i) {
                float diff = std::abs(h_out[i] - h_C_cpu[i]);
                if (diff > max_err) max_err = diff;
            }
            res.success = (max_err <= 1e-3f);
            std::cout << "[GPU - " << name << "] Kernel: " << std::fixed << std::setprecision(2) << res.kernel_time_ms 
                      << " ms (" << res.gflops << " GFLOPS, " << res.bandwidth_gb_s << " GB/s) | Total (w/ PCIe): " 
                      << res.total_time_ms << " ms | Verification Max Err: " << std::scientific << max_err 
                      << (res.success ? " [PASSED]" : " [FAILED]") << std::endl;
        } else {
            res.success = true;
            std::cout << "[GPU - " << name << "] Kernel: " << std::fixed << std::setprecision(2) << res.kernel_time_ms 
                      << " ms (" << res.gflops << " GFLOPS, " << res.bandwidth_gb_s << " GB/s) | Total (w/ PCIe): " 
                      << res.total_time_ms << " ms | Verification [SKIPPED]" << std::endl;
        }

        return res;
    };

    // ------------------------------------------------------------------------
    // Execute Benchmarks
    // ------------------------------------------------------------------------
    std::cout << "\n--- Running GPU Benchmarks ---" << std::endl;

    BenchmarkResult res_naive;
    if (!skip_naive) {
        res_naive = run_gpu_benchmark(k_naive, false, false, "Naive (Separate Activation)", h_C_gpu_naive);
    } else {
        std::cout << "[GPU - Naive (Separate Activation)] SKIPPED (Avoid TDR timeout)" << std::endl;
    }

    BenchmarkResult res_tiled = run_gpu_benchmark(k_tiled, false, true, "Tiled (Separate Activation)", h_C_gpu_tiled);

    BenchmarkResult res_naive_fused;
    if (!skip_naive) {
        res_naive_fused = run_gpu_benchmark(k_naive_fused, true, false, "Naive (Fused Activation)", h_C_gpu_naive_fused);
    } else {
        std::cout << "[GPU - Naive (Fused Activation)] SKIPPED (Avoid TDR timeout)" << std::endl;
    }

    BenchmarkResult res_tiled_fused = run_gpu_benchmark(k_tiled_fused, true, true, "Tiled (Fused Activation)", h_C_gpu_tiled_fused);

    // ------------------------------------------------------------------------
    // Device-to-Host D2H Bandwidth (Standalone measurement)
    // ------------------------------------------------------------------------
    std::cout << "\n[GPU] Benchmarking D2H Download..." << std::endl;
    auto t_d2h_start = std::chrono::high_resolution_clock::now();
    checkError(clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, bytesC, h_C_gpu_tiled_fused.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer D2H");
    clFinish(queue);
    auto t_d2h_end = std::chrono::high_resolution_clock::now();
    double d2h_time_ms = std::chrono::duration<double, std::milli>(t_d2h_end - t_d2h_start).count();
    double d2h_bandwidth = (double)(bytesC) / 1024.0 / 1024.0 / 1024.0 / (d2h_time_ms / 1000.0);
    std::cout << "[GPU] D2H Download Time:    " << d2h_time_ms << " ms (" << d2h_bandwidth << " GB/s)" << std::endl;

    // ------------------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------------------
    std::cout << "\n========================================================\n";
    std::cout << " Summary Comparison\n";
    std::cout << "========================================================\n";
    std::cout << std::left << std::setw(25) << "Implementation" 
              << std::right << std::setw(15) << "Time (ms)" 
              << std::right << std::setw(15) << "GFLOPS" 
              << std::right << std::setw(15) << "Speedup" << "\n";
    std::cout << "--------------------------------------------------------\n";
    
    if (validate) {
        std::cout << std::left << std::setw(25) << "CPU (Single-Thread)" 
                  << std::right << std::setw(15) << std::fixed << std::setprecision(2) << cpu_time_ms 
                  << std::right << std::setw(15) << std::fixed << std::setprecision(2) << (2.0 * M * N * K / (cpu_time_ms / 1000.0) / 1e9)
                  << std::right << std::setw(15) << "1.00x (Ref)" << "\n";
        
        auto print_speedup = [&](const char* name, const BenchmarkResult& res, bool skipped) {
            if (skipped) {
                std::cout << std::left << std::setw(25) << name 
                          << std::right << std::setw(15) << "SKIPPED" 
                          << std::right << std::setw(15) << "N/A"
                          << std::right << std::setw(15) << "N/A" << "\n";
                return;
            }
            double speedup = cpu_time_ms / res.kernel_time_ms;
            double speedup_total = cpu_time_ms / res.total_time_ms;
            std::cout << std::left << std::setw(25) << name 
                      << std::right << std::setw(15) << std::fixed << std::setprecision(2) << res.kernel_time_ms 
                      << std::right << std::setw(15) << std::fixed << std::setprecision(2) << res.gflops
                      << std::right << std::setw(15) << std::fixed << std::setprecision(2) << speedup << "x (Kernel)\n"
                      << std::left << std::setw(25) << ""
                      << std::right << std::setw(15) << std::fixed << std::setprecision(2) << res.total_time_ms
                      << std::right << std::setw(15) << ""
                      << std::right << std::setw(15) << std::fixed << std::setprecision(2) << speedup_total << "x (w/ PCIe)\n";
        };

        print_speedup("GPU Naive (Sep Act)", res_naive, skip_naive);
        print_speedup("GPU Tiled (Sep Act)", res_tiled, false);
        print_speedup("GPU Naive (Fused)", res_naive_fused, skip_naive);
        print_speedup("GPU Tiled (Fused)", res_tiled_fused, false);
    } else {
        auto print_no_speedup = [&](const char* name, const BenchmarkResult& res, bool skipped) {
            if (skipped) {
                std::cout << std::left << std::setw(25) << name 
                          << std::right << std::setw(15) << "SKIPPED" 
                          << std::right << std::setw(15) << "N/A"
                          << std::right << std::setw(15) << "N/A" << "\n";
                return;
            }
            std::cout << std::left << std::setw(25) << name 
                      << std::right << std::setw(15) << std::fixed << std::setprecision(2) << res.kernel_time_ms 
                      << std::right << std::setw(15) << std::fixed << std::setprecision(2) << res.gflops
                      << std::right << std::setw(15) << "N/A" << "\n";
        };
        print_no_speedup("GPU Naive (Sep Act)", res_naive, skip_naive);
        print_no_speedup("GPU Tiled (Sep Act)", res_tiled, false);
        print_no_speedup("GPU Naive (Fused)", res_naive_fused, skip_naive);
        print_no_speedup("GPU Tiled (Fused)", res_tiled_fused, false);
    }
    
    std::cout << "========================================================\n";

    // ------------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------------
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseKernel(k_naive);
    clReleaseKernel(k_tiled);
    clReleaseKernel(k_act);
    clReleaseKernel(k_naive_fused);
    clReleaseKernel(k_tiled_fused);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
