//
// Created by saleh on 17/05/24.
//

#include <gtest/gtest.h>
#include "testCommon.h"
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


inline bool test() {
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

    int blockSizes[] = {32, 64, 128, 256, 512, 1024 };
    goldCpu(tnOutGold, 0, tnIn, 0, tnWte, 0, tnWpe, 0, B, T, C);

    for(auto blockSize: blockSizes) {
        logger->info("Testing EncoderKernel with blockSize: {}", blockSize);

        kernels::EncoderKernel kernel(tnOut, 0, tnIn, 0, tnWte, 0, tnWpe, 0, B, T, C);
        logger->info("BlockSize: {}, Device Time: {} ns", blockSize, kernel.LaunchBlockingAndMeasureNanoSec(q, blockSize));

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