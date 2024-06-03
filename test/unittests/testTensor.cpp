//
// Created by saleh on 17/05/24.
//

#include <gtest/gtest.h>
#include "common/common.h"
#include "core/Tensor.h"
#include <sycl/sycl.hpp>

using namespace std;

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
    outQ = sycl::queue(sycl::default_selector_v, asycExceptionHandler, sycl::property::queue::enable_profiling());
    logger->info("SYCL queue initialized.");
    logger->info("Device Name: {}", outQ.get_device().get_info<sycl::info::device::name>());
    logger->info("Global Memory: {}", outQ.get_device().get_info<sycl::info::device::global_mem_size>());
    logger->info("Local Memory: {}", outQ.get_device().get_info<sycl::info::device::local_mem_size>());
    logger->info("CUs: {}", outQ.get_device().get_info<sycl::info::device::max_compute_units>());
}

TEST(tensor, Basic01) {
    logger->debug("Starting test Tensor.Basic01");
    sycl::queue q;
    initSycl(q);

    using namespace llmsycl::core;
    constexpr int SIZE = 64;

    Tensor<int> t1(q, {SIZE});
    for (int i = 0; i < t1.getSize(); i++) {
        t1.getHostBuffer()[i] = i;
    }
    t1.saveHostToNpy("/tmp/t1.npy");

    Tensor<int> t2 = Tensor<int>::loadToHost(q, "/tmp/t1.npy");
    for (int i = 0; i < t2.getSize(); i++) {
        EXPECT_EQ(t2.getHostBuffer()[i], i);
    }

    Tensor<int> t3(t2, false);
    for (int i = 0; i < t3.getSize(); i++) {
        EXPECT_EQ(t3.getHostBuffer()[i], i);
    }

    {
        t2.syncNonBlockingH2D();
        Tensor<int> t4(t2, true);
        for (int i = 0; i < t4.getSize(); i++) {
            EXPECT_EQ(t4.getHostBuffer()[i], i);
        }
    }

    Tensor<int> t4(t2, false);
    t1.syncNonBlockingH2D();
    t2.syncNonBlockingH2D();
    t3.syncNonBlockingH2D();
    t4.syncNonBlockingH2D();

    Tensor<int> t5(q, {SIZE});
    q.submit([&](sycl::handler &h) {
                 auto dt1 = t1.getDeviceBuffer();
                 auto dt2 = t2.getDeviceBuffer();
                 auto dt3 = t3.getDeviceBuffer();
                 auto dt4 = t4.getDeviceBuffer();
                 auto dt5 = t5.getDeviceBuffer();
                 h.parallel_for(t1.getSize(), [=](auto i) {
                     dt5[i] = dt1[i] + dt2[i] + dt3[i] + dt4[i];
                 });
             }
    );
    t5.syncNonBlockingD2H();
    q.wait();
    for (int i = 0; i < t5.getSize(); i++) {
        EXPECT_EQ(t5.getHostBuffer()[i], 4 * i);
    }
    t5.saveHostToNpy("/tmp/t5.npy");
    t5.saveHostToNpy(SIZE / 2, SIZE / 2, "/tmp/t5s.npy");

    Tensor<int> t6 = Tensor<int>::loadToHost(q, "/tmp/t5.npy");
    for (int i = 0; i < t6.getSize(); i++) {
        EXPECT_EQ(t6.getHostBuffer()[i], 4 * i);
    }
    Tensor<int> t7 = Tensor<int>::loadToHost(q, "/tmp/t5s.npy");
    for (int i = SIZE/2; i < t7.getSize(); i++) {
        EXPECT_EQ(t7.getHostBuffer()[i - SIZE/2], 4 * i);
    }

    t6.reshape({8, 8});
    t6.reshape({4, 2, 8});
    EXPECT_EQ(t6.reshape({4, 2, 0}), std::vector<size_t>({4, 2, 8}));
    EXPECT_EQ(t6.reshape({0, 8}), std::vector<size_t>({8, 8}));

}