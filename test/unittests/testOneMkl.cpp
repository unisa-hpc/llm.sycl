//
// Created by saleh on 24/05/24.
//


#include <gtest/gtest.h>
#include "common/common.h"
#include "core/Tensor.h"
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

using namespace std;
using namespace llmsycl;

// Create an exception handler for asynchronous SYCL exceptions


inline void initSycl(sycl::queue &outQ) {
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
    outQ = sycl::queue(sycl::cpu_selector_v, asycExceptionHandler, sycl::property::queue::enable_profiling());
    logger->info("SYCL queue initialized.");
    logger->info("Device Name: {}", outQ.get_device().get_info<sycl::info::device::name>());
    logger->info("Global Memory: {}", outQ.get_device().get_info<sycl::info::device::global_mem_size>());
    logger->info("Local Memory: {}", outQ.get_device().get_info<sycl::info::device::local_mem_size>());
    logger->info("CUs: {}", outQ.get_device().get_info<sycl::info::device::max_compute_units>());
}

TEST(oneMKL, gemm1) {
    sycl::queue queue;
    initSycl(queue);

    core::Tensor<float> tnA({16, 16});
    core::Tensor<float> tnB({16, 16});
    core::Tensor<float> tnC({16, 16});
    core::Tensor<float> tnGold({16, 16});
    float alpha = 1.5f;
    float beta = 0.0f;

    core::fillTensorWithRandomData(tnA);
    core::fillTensorWithRandomData(tnB);
    core::fillTensorWithRandomData(tnC);

    oneapi::mkl::transpose transA = oneapi::mkl::transpose::nontrans;
    oneapi::mkl::transpose transB = oneapi::mkl::transpose::nontrans;

    try {
        oneapi::mkl::blas::column_major::gemm(
                queue,
                transA,
                transB,
                16,
                16,
                16,
                alpha,
                tnB.getDeviceBuff(),  // swapped order
                16,
                tnA.getDeviceBuff(),  // swapped order
                16,
                beta,
                tnC.getDeviceBuff(),
                16
        );
        queue.wait_and_throw();
    } catch (std::exception &e) {
        logger->error("Exception: {}", e.what());
        std::terminate();
    }




    {
        int sizeM, sizeN, sizeK;
        sizeM = tnA.getShape()[0];
        sizeN = tnB.getShape()[1];
        sizeK = tnA.getShape()[1];
        auto accGold = tnGold.getAccessorHostReadWrite(0);
        auto accA = tnA.getAccessorHostRead(0);
        auto accB = tnB.getAccessorHostRead(0);
        auto accC = tnC.getAccessorHostRead(0);
        for (int j = 0; j < sizeM; j++) {
            for (int i = 0; i < sizeN; i++) {
                accGold[j * sizeN + i] = 0;
                for (int k = 0; k < sizeK; k++) {
                    accGold[j * sizeN + i] +=
                            alpha *accA[j * sizeK + k] *
                            accB[k * sizeN + i];
                }
                //accGold[j * sizeN + i] *= alpha;
                // beta is zero, no need to add beta*C
            }
        }
    }

    auto gold = tnGold.toVector();
    auto uut = tnC.toVector();
    // verify tnGold with tnC
    for (int i = 0; i < tnGold.getSize(); i++) {
        EXPECT_NEAR(gold[i], uut[i], 1e-5);
    }

}