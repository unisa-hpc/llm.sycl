//
// Created by saleh on 17/05/24.
//

#include <gtest/gtest.h>
#include "testCommon.h"
#include "common/common.h"
#include "core/Tensor.h"
#include "kernels/MatmulBias.h"
#include <sycl/sycl.hpp>

using namespace std;
using namespace llmsycl;


static inline void goldCpu(
        core::Tensor<float> &tnOut,
        size_t outOffset,
        const core::Tensor<float> &tnInput,
        size_t inputOffset,
        const core::Tensor<float> &tnWeight,
        size_t weightOffset,
        const core::Tensor<float> &tnBias,
        size_t biasOffset,
        int B, int T, int C, int OC,
        bool hasBias = true) {

    auto accTnOut = tnOut.getHostBuffer() + outOffset;
    auto accTnInput = tnInput.getHostBuffer() + inputOffset;
    auto accTnWeight = tnWeight.getHostBuffer() + weightOffset;
    auto accTnBias = tnBias.getHostBuffer() + biasOffset;

    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int o = 0; o < OC; o++) {
                float val = hasBias ? accTnBias[o] : 0.0f;
                for (int i = 0; i < C; i++) {
                    val += accTnInput[b * T * C + t * C + i] * accTnWeight[o * C + i];
                }
                accTnOut[b * T * OC + t * OC + o] = val;
            }
        }
    }
}

static inline void test() {
    sycl::queue q;
    prepareToTest(q);

    int B = 8;
    int T = 1024;
    int C = 768;
    int OC = 768 * 4; // expansion of 4, e.g. in the MLP


    // create tensors
    core::Tensor<float> tnOut(q, {(size_t) B * T * OC});
    core::Tensor<float> tnOutGold(q, {(size_t) B * T * OC});

    core::Tensor<float> tnIn(q, {(size_t) B * T * C});
    core::Tensor<float> tnWeight(q, {(size_t) OC * C});
    core::Tensor<float> tnBias(q, {(size_t) OC});

    core::fillTensorWithRandomData(tnIn);
    core::fillTensorWithRandomData(tnWeight);
    core::fillTensorWithRandomData(tnBias);

    int blockSizes[] = {4, 8, 16, 32};
    goldCpu(
            tnOutGold, 0,
            tnIn, 0,
            tnWeight, 0,
            tnBias, 0,
            B, T, C, OC
    );

    for (auto blockSize: blockSizes) {
        auto testCase = [&](bool hasBiasOpt) -> bool {
            logger->info("Testing MatmulBiasKernel with blockSize: {} and hasBias: {}", blockSize, hasBiasOpt);

            kernels::MatmulBias kernel(
                    tnOut.getDeviceBuffer(),
                    tnIn.getDeviceBuffer(),
                    tnWeight.getDeviceBuffer(),
                    tnBias.getDeviceBuffer(),
                    B, T, C, OC,
                    true
            );

            logger->info("BlockSize: {}, Device Time: {} ns", blockSize,
                         kernel.LaunchBlockingAndMeasureNanoSec(q, blockSize, {}));

            tnOut.syncBlockingD2H();

            auto accTnOut = tnOut.getHostBuffer();
            auto accTnOutGold = tnOutGold.getHostBuffer();
            for (int i = 0; i < B * T * OC; i++) {
                /// TODO: This is a very loose verification test. We need to improve it.
                if (std::abs(accTnOut[i] - accTnOutGold[i]) > 1) {
                    logger->error("\tMatmulBiasKernel tnOut failed the verification test against the gold at index: {}",
                                  i);
                    return false;
                }
            }
            return true;
        };

        EXPECT_TRUE(testCase(true));
        EXPECT_TRUE(testCase(false));

    }
}

TEST(kernelMatmulBias, basic01) {
    test();
}