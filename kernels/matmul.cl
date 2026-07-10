#ifndef TS
#define TS 16
#endif

// Naive matrix multiplication: C = A * B
__kernel void matmul_naive(
    const int M, const int N, const int K,
    __global const float* A,
    __global const float* B,
    __global float* C)
{
    int r = get_global_id(0); // Row index (0 to M-1)
    int c = get_global_id(1); // Col index (0 to N-1)

    if (r < M && c < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[r * K + k] * B[k * N + c];
        }
        C[r * N + c] = sum;
    }
}

// Tiled matrix multiplication using local memory
__kernel void matmul_tiled(
    const int M, const int N, const int K,
    __global const float* A,
    __global const float* B,
    __global float* C)
{
    // Local memory shared within the workgroup
    __local float tileA[TS][TS];
    __local float tileB[TS][TS];

    // Thread coordinates
    int global_r = get_global_id(0); // Row in C
    int global_c = get_global_id(1); // Col in C
    int local_r = get_local_id(0);   // Row within workgroup
    int local_c = get_local_id(1);   // Col within workgroup

    float sum = 0.0f;
    int numTiles = (K + TS - 1) / TS;

    for (int t = 0; t < numTiles; ++t) {
        // Load tile from A. Check boundaries.
        int a_col = t * TS + local_c;
        if (global_r < M && a_col < K) {
            tileA[local_r][local_c] = A[global_r * K + a_col];
        } else {
            tileA[local_r][local_c] = 0.0f;
        }

        // Load tile from B. Check boundaries.
        int b_row = t * TS + local_r;
        if (b_row < K && global_c < N) {
            tileB[local_r][local_c] = B[b_row * N + global_c];
        } else {
            tileB[local_r][local_c] = 0.0f;
        }

        // Wait for all threads in the workgroup to finish loading the tiles
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform partial multiplication
        for (int k = 0; k < TS; ++k) {
            sum += tileA[local_r][k] * tileB[k][local_c];
        }

        // Synchronize before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result to global memory
    if (global_r < M && global_c < N) {
        C[global_r * N + global_c] = sum;
    }
}

// Standalone Activation Kernel
__kernel void apply_activation(
    const int M, const int N,
    __global float* C,
    const int activation_type) // 0: None, 1: ReLU, 2: Sigmoid
{
    int r = get_global_id(0);
    int c = get_global_id(1);

    if (r < M && c < N) {
        int idx = r * N + c;
        float val = C[idx];
        if (activation_type == 1) { // ReLU
            C[idx] = val > 0.0f ? val : 0.0f;
        } else if (activation_type == 2) { // Sigmoid
            C[idx] = 1.0f / (1.0f + exp(-val));
        }
    }
}

// Fused Naive MatMul + Activation Kernel
__kernel void matmul_naive_fused(
    const int M, const int N, const int K,
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int activation_type) // 0: None, 1: ReLU, 2: Sigmoid
{
    int r = get_global_id(0);
    int c = get_global_id(1);

    if (r < M && c < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[r * K + k] * B[k * N + c];
        }
        
        // Fused Activation
        if (activation_type == 1) { // ReLU
            sum = sum > 0.0f ? sum : 0.0f;
        } else if (activation_type == 2) { // Sigmoid
            sum = 1.0f / (1.0f + exp(-sum));
        }
        
        C[r * N + c] = sum;
    }
}

// Fused Tiled MatMul + Activation Kernel
__kernel void matmul_tiled_fused(
    const int M, const int N, const int K,
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int activation_type) // 0: None, 1: ReLU, 2: Sigmoid
{
    __local float tileA[TS][TS];
    __local float tileB[TS][TS];

    int global_r = get_global_id(0);
    int global_c = get_global_id(1);
    int local_r = get_local_id(0);
    int local_c = get_local_id(1);

    float sum = 0.0f;
    int numTiles = (K + TS - 1) / TS;

    for (int t = 0; t < numTiles; ++t) {
        int a_col = t * TS + local_c;
        if (global_r < M && a_col < K) {
            tileA[local_r][local_c] = A[global_r * K + a_col];
        } else {
            tileA[local_r][local_c] = 0.0f;
        }

        int b_row = t * TS + local_r;
        if (b_row < K && global_c < N) {
            tileB[local_r][local_c] = B[b_row * N + global_c];
        } else {
            tileB[local_r][local_c] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; ++k) {
            sum += tileA[local_r][k] * tileB[k][local_c];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_r < M && global_c < N) {
        if (activation_type == 1) { // ReLU
            sum = sum > 0.0f ? sum : 0.0f;
        } else if (activation_type == 2) { // Sigmoid
            sum = 1.0f / (1.0f + exp(-sum));
        }
        C[global_r * N + global_c] = sum;
    }
}
