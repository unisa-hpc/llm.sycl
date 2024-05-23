//
// Created by saleh on 17/05/24.
//

#include <gtest/gtest.h>
#include "common/common.h"
#include "core/Tensor.h"
#include "kernels/Encoder.h"
#include <sycl/sycl.hpp>

using namespace std;
using namespace llmsycl;

void goldCpu(
        core::Tensor<float> &tnOut,
        size_t outOffset,
        const core::Tensor<int> &tnIn,
        size_t inOffset,
        const core::Tensor<float> &tnWte,
        size_t wteOffset,
        const core::Tensor<float> &tnWpe,
        size_t wpeOffset,
        int B, int T, int C) {
    auto accTnOut = tnOut.getAccessorHostWrite(outOffset);
    auto accTnIn = tnIn.getAccessorHostRead(inOffset);
    auto accTnWte = tnWte.getAccessorHostRead(wteOffset);
    auto accTnWpe = tnWpe.getAccessorHostRead(wpeOffset);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int ix = accTnIn[b * T + t];
            for (int i = 0; i < C; i++) {
                accTnOut[b * T * C + t * C + i] = accTnWte[ix * C + i] + accTnWpe[t * C + i];
            }
        }
    }
}

void prepareToTest(sycl::queue &outQ) {
    auto asycExceptionHandler = [](sycl::exception_list e_list) {
        for (std::exception_ptr const &e: e_list) {
            try {
                std::rethrow_exception(e);
            }
            catch (std::exception const &e) {
                logger->error("Failure: {}", e.what());
                std::terminate();
            }
        }
    };
    outQ = sycl::queue(sycl::gpu_selector_v, asycExceptionHandler, sycl::property::queue::enable_profiling());
    logger->info("SYCL queue initialized.");
    logger->info("Device Name: {}", outQ.get_device().get_info<sycl::info::device::name>());
    logger->info("Global Memory: {}", outQ.get_device().get_info<sycl::info::device::global_mem_size>());
    logger->info("Local Memory: {}", outQ.get_device().get_info<sycl::info::device::local_mem_size>());
    logger->info("CUs: {}", outQ.get_device().get_info<sycl::info::device::max_compute_units>());
}

bool test() {
    sycl::queue q;
    prepareToTest(q);

    int B = 8;
    int T = 1024;
    int C = 768;
    int V = 50257;

    // create tensors
    core::Tensor<float> tnOut({(size_t) B * T * C});
    core::Tensor<float> tnOutGold({(size_t) B * T * C});
    core::Tensor<int> tnIn({(size_t) B * T});
    core::Tensor<float> tnWte({(size_t) V * C});
    core::Tensor<float> tnWpe({(size_t) T * C});

    core::fillTensorWithRandomData(tnIn, V);
    core::fillTensorWithRandomData(tnWte);
    core::fillTensorWithRandomData(tnWpe);

    int blockSizes[] = {32 };
    goldCpu(tnOutGold, 0, tnIn, 0, tnWte, 0, tnWpe, 0, B, T, C);

    for(auto blockSize: blockSizes) {
        logger->info("Testing EncoderKernel with blockSize: {}", blockSize);

        kernels::EncoderKernel kernel(tnOut, 0, tnIn, 0, tnWte, 0, tnWpe, 0, B, T, C);

        kernel.Launch(q, blockSize);
        q.wait_and_throw();
        logger->info("Got past q.wait_and_throw()");

        auto accTnOut = tnOut.getAccessorHostRead();
        auto accTnOutGold = tnOutGold.getAccessorHostRead();
        for (int i = 0; i < B * T * C; i++) {
            if (std::abs(accTnOut[i] - accTnOutGold[i]) > 1e-5) {
                logger->error("\tEncoderKernel failed the verification test against the gold at index: {}", i);
                return false;
            }
        }
    }

    return true;
}

TEST(kernelEncode, basic01) {
    test();
}