//
// Created by saleh on 17/05/24.
//

#include <gtest/gtest.h>
#include "common/common.h"
#include "core/tensor.h"
#include <sycl/sycl.hpp>

using namespace std;

// Create an exception handler for asynchronous SYCL exceptions


void initSycl(sycl::queue &outQ) {
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
    logger->debug("Starting test tensor.Basic01");
    sycl::queue q;
    initSycl(q);

    using namespace llmsycl::core;
    constexpr int SIZE = 64;

    tensor<int> t1({SIZE});
    for (int i = 0; i < t1.getSize(); i++) {
        t1.getAccessorHostWrite()[i] = i;
    }
    t1.save("/tmp/t1.npy");

    tensor<int> t2 = tensor<int>::load("/tmp/t1.npy");
    for (int i = 0; i < t2.getSize(); i++) {
        EXPECT_EQ(t2.getAccessorHostWrite()[i], i);
    }

    tensor<int> t3(q, t2, false);
    for (int i = 0; i < t3.getSize(); i++) {
        EXPECT_EQ(t3.getAccessorHostWrite()[i], i);
    }

    tensor<int> t4(q, t2, true);
    for (int i = 0; i < t4.getSize(); i++) {
        EXPECT_EQ(t4.getAccessorHostWrite()[i], i);
    }

    tensor<int> t5({SIZE});
    q.submit([&](sycl::handler &h) {
                 sycl::accessor dt1 = t1.getAccessorDeviceRead(h);
                 sycl::accessor dt2 = t2.getAccessorDeviceRead(h);
                 sycl::accessor dt3 = t3.getAccessorDeviceRead(h);
                 sycl::accessor dt4 = t4.getAccessorDeviceRead(h);
                 sycl::accessor dt5 = t5.getAccessorDeviceWrite(h);

                 h.parallel_for(t1.getSize(), [=](auto i) {
                     dt5[i] = dt1[i] + dt2[i] + dt3[i] + dt4[i];
                 });
             }
    );
    q.wait();
    for (int i = 0; i < t5.getSize(); i++) {
        EXPECT_EQ(t5.getAccessorHostWrite()[i], 4*i);
    }
    t5.save("/tmp/t5.npy");

    tensor<int> t6 = tensor<int>::load("/tmp/t5.npy");
    for (int i = 0; i < t5.getSize(); i++) {
        EXPECT_EQ(t6.getAccessorHostWrite()[i], 4*i);
    }

    t6.reshape({8, 8});
    t6.reshape({4, 2, 8});
    EXPECT_EQ(t6.reshape({4, 2, 0}), std::vector<size_t>({4, 2, 8}));
    EXPECT_EQ(t6.reshape({0, 8}), std::vector<size_t>({8, 8}));
}