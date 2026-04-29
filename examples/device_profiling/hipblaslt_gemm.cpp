/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <chrono>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <iomanip>
#include <iostream>
#include <thread>

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                                                 \
    if (error != hipSuccess)                                                   \
    {                                                                          \
        fprintf(stderr, "Hip error: '%s'(%d) at %s:%d\n",                      \
                hipGetErrorString(error), error, __FILE__, __LINE__);          \
        exit(EXIT_FAILURE);                                                    \
    }
#endif

#ifndef CHECK_HIPBLASLT_ERROR
#define CHECK_HIPBLASLT_ERROR(error)                                           \
    if (error != HIPBLAS_STATUS_SUCCESS)                                       \
    {                                                                          \
        fprintf(stderr, "hipBLASLt error(Err=%d) at %s:%d\n", error, __FILE__, \
                __LINE__);                                                     \
        fprintf(stderr, "\n");                                                 \
        exit(EXIT_FAILURE);                                                    \
    }
#endif

int main(int argc, char *argv[])
{

    int dev_count;
    CHECK_HIP_ERROR(hipGetDeviceCount(&dev_count));
    printf("HIP devices: %d\n", dev_count);

    if (argc != 4)
    {
        std::cout << "Usage: ./hipblas_gemm <m> <n> <k>" << std::endl;
        return -1;
    }
    int64_t m = atoi(argv[1]), n = atoi(argv[2]), k = atoi(argv[3]);

    /** This is a NN example with
     *  a = (m, k). lda = m
     *  b = (k, n). ldb = k
     *  c = d = (m, n). ldc = ldd = m
     */
    int64_t batch_count = 1;
    float alpha = 1.0;
    float beta = 0.0;

    void *a0, *b0, *c0, *d0, *alphaVec0;           // host
    void *d_a0, *d_b0, *d_c0, *d_d0, *d_alphaVec0; // device
    void *a1, *b1, *c1, *d1, *alphaVec1;           // host
    void *d_a1, *d_b1, *d_c1, *d_d1, *d_alphaVec1; // device

    void *d_workspace0, *d_workspace1;
    int64_t max_workspace_size = 32 * 1024 * 1024;

    hipStream_t stream;
    hipblasLtHandle_t handle;

    CHECK_HIP_ERROR(hipStreamCreate(&stream));
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));
    CHECK_HIP_ERROR(
        hipMalloc(&d_a0, m * k * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(
        hipMalloc(&d_a1, m * k * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(
        hipMalloc(&d_b0, n * k * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(
        hipMalloc(&d_b1, n * k * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(
        hipMalloc(&d_c0, m * n * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(
        hipMalloc(&d_c1, m * n * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(
        hipMalloc(&d_d0, m * n * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(
        hipMalloc(&d_d1, m * n * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(hipMalloc(&d_alphaVec0, m * batch_count * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_alphaVec1, m * batch_count * sizeof(float)));

    CHECK_HIP_ERROR(
        hipHostMalloc(&a0, m * k * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(
        hipHostMalloc(&a1, m * k * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(
        hipHostMalloc(&b0, n * k * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(
        hipHostMalloc(&b1, n * k * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(
        hipHostMalloc(&c0, m * n * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(
        hipHostMalloc(&c1, m * n * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(
        hipHostMalloc(&d0, m * n * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(
        hipHostMalloc(&d1, m * n * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(hipHostMalloc(&alphaVec0, m * batch_count * sizeof(float)));
    CHECK_HIP_ERROR(hipHostMalloc(&alphaVec1, m * batch_count * sizeof(float)));

    if (max_workspace_size > 0)
    {
        CHECK_HIP_ERROR(hipMalloc(&d_workspace0, max_workspace_size));
        CHECK_HIP_ERROR(hipMalloc(&d_workspace1, max_workspace_size));
    }

    for (int i = 0; i < m * k * batch_count; i++)
    {
        ((hipblasLtHalf *)a0)[i] = static_cast<hipblasLtHalf>((rand() % 7) - 3);
        ((hipblasLtHalf *)a1)[i] = static_cast<hipblasLtHalf>((rand() % 7) - 3);
    }
    for (int i = 0; i < n * k * batch_count; i++)
    {
        ((hipblasLtHalf *)b0)[i] = static_cast<hipblasLtHalf>((rand() % 7) - 3);
        ((hipblasLtHalf *)b1)[i] = static_cast<hipblasLtHalf>((rand() % 7) - 3);
    }
    for (int i = 0; i < m * n * batch_count; i++)
    {
        ((hipblasLtHalf *)c0)[i] = static_cast<hipblasLtHalf>((rand() % 7) - 3);
        ((hipblasLtHalf *)c1)[i] = static_cast<hipblasLtHalf>((rand() % 7) - 3);
    }
    for (int i = 0; i < m * batch_count; ++i)
    {
        ((float *)alphaVec0)[i] = static_cast<float>((rand() % 7) - 3);
        ((float *)alphaVec1)[i] = static_cast<float>((rand() % 7) - 3);
    }

    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, m, k, m));
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, k, n, k));
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, m, n, m));
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatrixLayoutCreate(&matD, HIP_R_16F, m, n, m));

    if (batch_count > 1)
    {
        int64_t stride_a = m * k;
        int64_t stride_b = k * n;
        int64_t stride_c = m * n;
        int64_t stride_d = m * n;
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
            sizeof(batch_count)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matA, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_a,
            sizeof(stride_a)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
            sizeof(batch_count)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matB, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_b,
            sizeof(stride_b)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matC, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
            sizeof(batch_count)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matC, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_c,
            sizeof(stride_c)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matD, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
            sizeof(batch_count)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matD, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_d,
            sizeof(stride_d)));
    }

    hipblasLtMatmulDesc_t matmul;
    hipblasOperation_t trans_a = HIPBLAS_OP_N, trans_b = HIPBLAS_OP_N;
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));

    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    // Set User Preference attributes
    hipblasLtMatmulPreference_t pref;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_size,
        sizeof(max_workspace_size)));

    const int request_solutions = 1;
    hipblasLtMatmulHeuristicResult_t heuristicResult[request_solutions];
    int returnedAlgoCount = 0;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(
        handle, matmul, matA, matB, matC, matD, pref, request_solutions,
        heuristicResult, &returnedAlgoCount));

    if (returnedAlgoCount == 0)
    {
        std::cerr << "No valid solution found!" << std::endl;
        return -1;
    }

    uint64_t workspace_size = 0;
    for (int i = 0; i < returnedAlgoCount; i++)
    {
        uint64_t h_workspace_size = heuristicResult[i].workspaceSize;
        std::cout << "Algorithm requires workspace size: " << h_workspace_size
                  << std::endl;
        workspace_size = max(workspace_size, h_workspace_size);
    }

    // In this sample, the workspace is already allocated with
    // max_workspace_size If not, allocate d_workspace here
    // CHECK_HIP_ERRORhipMalloc(&d_workspace, workspace_size));

    auto ROI = [&](bool buff)
    {
        if (buff)
        {
            CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(
                handle, matmul, &alpha, d_a0, matA, d_b0, matB, &beta, d_c0,
                matC, d_d0, matD, &heuristicResult[0].algo, d_workspace0,
                workspace_size, stream));
        }
        else
        {
            CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(
                handle, matmul, &alpha, d_a1, matA, d_b1, matB, &beta, d_c1,
                matC, d_d1, matD, &heuristicResult[0].algo, d_workspace1,
                workspace_size, stream));
        }
    };
    CHECK_HIP_ERROR(hipSetDevice(0));
    constexpr int standalone_iters = 200;
    // Initial sleep to let profiler establish baseline
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    for (int i = 0; i < standalone_iters; i++)
    {
        ROI(false);
        CHECK_HIP_ERROR(hipDeviceSynchronize());
        // Sleep between kernels so profiler can sample during idle gap
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    // CHECK_HIP_ERROR(hipDeviceSynchronize());
    // hipEvent_t start, stop;
    // CHECK_HIP_ERROR(hipEventCreate(&start));
    // CHECK_HIP_ERROR(hipEventCreate(&stop));
    // CHECK_HIP_ERROR(hipEventRecord(start, stream));

    // CHECK_HIP_ERROR(hipDeviceSynchronize());
    // auto start_time = std::chrono::steady_clock::now();
    // ROI(false);
    // CHECK_HIP_ERROR(hipDeviceSynchronize());
    // auto stop_time = std::chrono::steady_clock::now();
    // // get last execution time
    // CHECK_HIP_ERROR(hipGetLastError());
    // CHECK_HIP_ERROR(hipEventRecord(stop, stream));
    // CHECK_HIP_ERROR(hipEventSynchronize(start));
    // CHECK_HIP_ERROR(hipEventSynchronize(stop));
    // float kernel_time;
    // CHECK_HIP_ERROR(hipEventElapsedTime(&kernel_time, start, stop));
    // CHECK_HIP_ERROR(hipEventDestroy(start));
    // CHECK_HIP_ERROR(hipEventDestroy(stop));

    // auto cpu_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
    //                        stop_time - start_time)
    //                        .count() /
    //                    1000000.0;
    // // std::cout << "hipblaslt gemm" << std::endl
    // //           << "m, n, k = " << m << ", " << n << ", " << k
    // //           << " hip exec time: " << std::setprecision(6) << kernel_time
    // //           << " cpu exec time: " << std::setprecision(6) << cpu_elapsed
    // //           << std::endl;
    // std::cout << std::setprecision(6) << cpu_elapsed << std::endl;

    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
    CHECK_HIP_ERROR(hipFree(d_workspace0));
    CHECK_HIP_ERROR(hipFree(d_workspace1));
    CHECK_HIP_ERROR(hipFree(a0));
    CHECK_HIP_ERROR(hipFree(a1));
    CHECK_HIP_ERROR(hipFree(b0));
    CHECK_HIP_ERROR(hipFree(b1));
    CHECK_HIP_ERROR(hipFree(c0));
    CHECK_HIP_ERROR(hipFree(c1));
    CHECK_HIP_ERROR(hipFree(d0));
    CHECK_HIP_ERROR(hipFree(d1));
    CHECK_HIP_ERROR(hipFree(alphaVec0));
    CHECK_HIP_ERROR(hipFree(alphaVec1));
    CHECK_HIP_ERROR(hipFree(d_a0));
    CHECK_HIP_ERROR(hipFree(d_a1));
    CHECK_HIP_ERROR(hipFree(d_b0));
    CHECK_HIP_ERROR(hipFree(d_b1));
    CHECK_HIP_ERROR(hipFree(d_c0));
    CHECK_HIP_ERROR(hipFree(d_c1));
    CHECK_HIP_ERROR(hipFree(d_d0));
    CHECK_HIP_ERROR(hipFree(d_d1));
    CHECK_HIP_ERROR(hipFree(d_alphaVec0));
    CHECK_HIP_ERROR(hipFree(d_alphaVec1));
    CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
    return 0;
}
