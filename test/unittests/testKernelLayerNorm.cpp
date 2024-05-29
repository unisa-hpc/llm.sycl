//
// Created by saleh on 17/05/24.
//

#include <gtest/gtest.h>
#include "testCommon.h"
#include "common/common.h"
#include "core/Tensor.h"
#include "kernels/LayerNorm.h"
#include <sycl/sycl.hpp>

using namespace std;
using namespace llmsycl;

/*
 * float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
 * */
void goldCpu(
        core::Tensor<float> &tnOut,
        size_t outOffset,
        core::Tensor<float> &tnMean,
        size_t meanOffset,
        core::Tensor<float> &tnRstd,
        size_t rstdOffset,
        const core::Tensor<float> &tnInp,
        size_t inpOffset,
        const core::Tensor<float> &tnWeight,
        size_t weightOffset,
        const core::Tensor<float> &tnBias,
        size_t biasOffset,
        int B, int T, int C) {

    auto accTnOut = tnOut.getAccessorHostWrite(outOffset);
    auto accTnMean = tnMean.getAccessorHostWrite(meanOffset);
    auto accTnRstd = tnRstd.getAccessorHostWrite(rstdOffset);
    auto accTnInp = tnInp.getAccessorHostRead(inpOffset);
    auto accTnWeight = tnWeight.getAccessorHostRead(weightOffset);
    auto accTnBias = tnBias.getAccessorHostRead(biasOffset);

    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += accTnInp[b * T * C + t * C + i];
            }
            m = m / C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = accTnInp[b * T * C + t * C + i] - m;
                v += xshift * xshift;
            }
            v = v / C;
            // calculate the rstd
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]

            for (int i = 0; i < C; i++) {
                float n = (s * (accTnInp[b * T * C + t * C + i] - m)); // normalized output
                float o = n * accTnWeight[i] + accTnBias[i]; // scale and shift it
                accTnOut[b * T * C + t * C + i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            accTnMean[b * T + t] = m;
            accTnRstd[b * T + t] = s;
        }
    }

}

inline bool test() {
    sycl::queue q;
    prepareToTest(q);

    int B = 8;
    int T = 1024;
    int C = 768;

    // create tensors
    core::Tensor<float> tnOut({(size_t) B * T * C});
    core::Tensor<float> tnOutGold({(size_t) B * T * C});

    core::Tensor<float> tnMean({(size_t) B * T});
    core::Tensor<float> tnMeanGold({(size_t) B * T});

    core::Tensor<float> tnRstd({(size_t) B * T});
    core::Tensor<float> tnRstdGold({(size_t) B * T});

    core::Tensor<float> tnIn({(size_t) B * T * C});
    core::Tensor<float> tnWeight({(size_t) C});
    core::Tensor<float> tnBias({(size_t) C});

    core::fillTensorWithRandomData(tnIn);
    core::fillTensorWithRandomData(tnWeight);
    core::fillTensorWithRandomData(tnBias);

    int blockSizes[] = {32, 64, 128, 256, 512, 1024};
    goldCpu(
            tnOutGold, 0,
            tnMeanGold, 0,
            tnRstdGold, 0,
            tnIn, 0,
            tnWeight, 0,
            tnBias, 0,
            B, T, C
    );

    for (auto blockSize: blockSizes) {
        logger->info("Testing EncoderKernel with blockSize: {}", blockSize);

        kernels::LayerNorm kernel(
                tnOut, 0,
                tnMean, 0,
                tnRstd, 0,
                tnIn, 0,
                tnWeight, 0,
                tnBias, 0,
                B, T, C
        );

        logger->info("BlockSize: {}, Device Time: {} ns", blockSize,
                     kernel.LaunchBlockingAndMeasureNanoSec(q, blockSize));

        auto accTnOut = tnOut.getAccessorHostRead();
        auto accTnOutGold = tnOutGold.getAccessorHostRead();
        for (int i = 0; i < B * T * C; i++) {
            if (std::abs(accTnOut[i] - accTnOutGold[i]) > 1e-5) {
                logger->error("\tLayerNormKernel tnOut failed the verification test against the gold at index: {}", i);
                return false;
            }
        }


        auto accTnMean = tnMean.getAccessorHostRead();
        auto accTnMeanGold = tnMeanGold.getAccessorHostRead();
        for (int i = 0; i < B * T; i++) {
            if (std::abs(accTnMean[i] - accTnMeanGold[i]) > 1e-5) {
                logger->error("\tLayerNormKernel tnMean failed the verification test against the gold at index: {}", i);
                return false;
            }
        }

        auto accTnRstd = tnRstd.getAccessorHostRead();
        auto accTnRstdGold = tnRstdGold.getAccessorHostRead();
        for (int i = 0; i < B * T; i++) {
            if (std::abs(accTnRstd[i] - accTnRstdGold[i]) > 1e-5) {
                logger->error("\tLayerNormKernel tnRstd failed the verification test against the gold at index: {}", i);
                return false;
            }
        }
    }

    return true;
}

TEST(kernelLayerNorm, basic01) {
    EXPECT_TRUE(test());
}