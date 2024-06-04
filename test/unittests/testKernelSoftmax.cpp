//
// Created by saleh on 17/05/24.
//

#include <gtest/gtest.h>
#include <sycl/sycl.hpp>

#include "testCommon.h"
#include "common/common.h"
#include "core/Tensor.h"
#include "kernels/Softmax.h"

using namespace std;
using namespace llmsycl;


static inline void cpuGold(
        core::Tensor<float> &tnOut,
        const core::Tensor<float> &tnIn,
        int N, int C) {
    // online version of softmax on CPU from the paper "Online normalizer calculation for softmax"
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    auto inp = tnIn.getHostBuffer();
    auto out = tnOut.getHostBuffer();

    for (int i = 0; i < N; i++) {
        float maxval = -INFINITY;
        float sum = 0.0f;
        for (int j = 0; j < C; j++) {
            float maxval_prev = maxval;
            if (inp[i * C + j] > maxval) {
                maxval = inp[i * C + j];
                sum = sum * std::exp(maxval_prev - maxval) + std::exp(inp[i * C + j] - maxval);
            } else {
                sum += std::exp(inp[i * C + j] - maxval);
            }
        }

        for (int j = 0; j < C; j++) {
            out[i * C + j] = std::exp(inp[i * C + j] - maxval) / sum;
        }
    }
}

static inline void cpuGold2(
        core::Tensor<float> &tnOut,
        const core::Tensor<float> &tnIn,
        float invTemperature,
        int N, int C) {
    // online version of softmax on CPU from the paper "Online normalizer calculation for softmax"
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    auto inp = tnIn.getHostBuffer();
    auto out = tnOut.getHostBuffer();

    for (size_t i = 0; i < N; ++i) {
        double max_val = -INFINITY;
        double sum_exp = 0.0;

        // Find the maximum value in the current row
        for (size_t j = 0; j < C; ++j) {
            if (inp[i * C + j] > max_val) {
                max_val = inp[i * C + j];
            }
        }

        // Compute the sum of exponentials after subtracting the maximum value
        for (size_t j = 0; j < C; ++j) {
            double exp_val = exp((inp[i * C + j] - max_val) * invTemperature);
            out[i * C + j] = exp_val;
            sum_exp += exp_val;
        }

        // Normalize the probabilities
        for (size_t j = 0; j < C; ++j) {
            out[i * C + j] /= sum_exp;
        }
    }
}

static inline void cpuGold3(
        core::Tensor<float> &tnOut,
        const core::Tensor<float> &tnIn,
        float invTemperature,
        int N, int C) {
    // online version of softmax on CPU from the paper "Online normalizer calculation for softmax"
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    auto inp = tnIn.getHostBuffer();
    auto out = tnOut.getHostBuffer();

    for (size_t i = 0; i < N; ++i) {
        // const size_t subgroup_size = item.get_sub_group().get_local_range().get(0); // Get the size of the subgroup
        // const size_t num_sg_in_wg = item.get_local_range(0) / subgroup_size; // num warps in the workgroup
        // const size_t local_id = item.get_local_id(0); // Get the local ID of the work-item
        // const size_t subgroup_index = local_id / subgroup_size; // Calculate the index of the subgroup
        // const size_t idInSg = local_id % subgroup_size; // Calculate the index of the work-item within the subgroup

        // inp is (N, T)
        // out is (N, T), each row of inp will get softmaxed
        if (i < N) {
            float maxval = -INFINITY;
            double sum = 0.0;
            for (int j = 0; j < C; j++) {
                if (inp[i*C+j] > maxval) {
                    maxval = inp[i*C+j];
                }
            }
            for (int j = 0; j < C; j++) {
                sum += std::exp((inp[i*C+j] - maxval) * invTemperature);
                out[i*C+j] = (float) sum;
            }
            auto sumInverse = sum == 0.0 ?
                              0.0 :
                              1.0/sum;

            for (int j = 0; j < C; j++) {
                out[i*C+j] *= (j > i/C) ? 0 : out[i*C+j] * (float)sumInverse;
            }
        }
    }
}

static inline bool test() {
    sycl::queue q;
    prepareToTest(q);
    int B = 8;
    int T = 1024;
    int C = 768;
    float invTemperature = 2.567f;

    core::Tensor<float> tnOutGold(q, {(size_t) B * T * C});
    core::Tensor<float> tnOut(q, {(size_t) B * T * C});
    core::Tensor<float> tnIn(q, {(size_t) B * T * C});
    core::fillTensorWithRandomData(tnIn);

    cpuGold3(tnOutGold, tnIn, invTemperature, B * T, C);

    llmsycl::kernels::Softmax kernel(
            tnOut.getDeviceBuffer(),
            tnIn.getDeviceBuffer(),
            invTemperature,
            B*T, C
    );

    int blockSizes[] = {32, 64, 128, 256, 512, 1024};
    for (auto blockSize: blockSizes) {
        logger->info("Testing SoftmaxKernel with blockSize: {}", blockSize);
        logger->info("BlockSize: {}, Device Time: {} ns", blockSize,
                     kernel.LaunchBlockingAndMeasureNanoSec(q, blockSize));

        tnOut.syncBlockingD2H();
        auto accTnOut = tnOut.getHostBuffer();
        auto accTnOutGold = tnOutGold.getHostBuffer();
        for (int i = 0; i < B * T * C; i++) {
            if (std::abs(accTnOut[i] - accTnOutGold[i]) > 1e-1) {
                logger->error("\tSoftmaxKernel failed the verification test against the gold at index: {}", i);
                logger->error("\t\tExpected: {}, Got: {}", accTnOutGold[i], accTnOut[i]);

                return false;
            }
        }

    }
    return true;
}

TEST(kernelSoftmax, basic01) {
    EXPECT_TRUE(test());
}