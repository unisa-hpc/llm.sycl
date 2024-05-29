//
// Created by saleh on 23/05/24.
//

#pragma once

#include <sycl/sycl.hpp>

#include "core/Tensor.h"
#include "BaseKernel.h"

namespace llmsycl::kernels {

    class Gelu : public BaseKernel {
        friend class sycl::handler;

    public:
        Gelu(
                core::Tensor<float> &tnOutput,
                size_t outputOffset,
                core::Tensor<float> &tnInput,
                size_t inputOffset,
                int N) :
                BaseKernel("Gelu"),
                tnOutput(tnOutput), outputOffset(outputOffset),
                tnInput(tnInput), inputOffset(inputOffset),
                N(N), geluScalingFactor(std::sqrt(2.0f / M_PI)) {

            addTensorDetailsToReport("tnOutput", tnOutput);
            addTensorDetailsToReport("tnInput", tnInput);
            addScalarParamToReport("outputOffset", outputOffset);
            addScalarParamToReport("inputOffset", inputOffset);
            addScalarParamToReport("N", N);
        }

        sycl::event Launch(sycl::queue &q, int blockSize) override {
            auto event = q.submit([&](sycl::handler &h) {
                auto accTnOutput = tnOutput.getAccessorDeviceWrite(h, outputOffset);
                auto accTnInput = tnInput.getAccessorDeviceRead(h, inputOffset);
                const size_t capturedN = this->N;
                const auto capturedGeluScalingFactor = this->geluScalingFactor;

                h.parallel_for(
                        sycl::nd_range<1>(
                                sycl::range<1>(Helpers::MakeDivisible(N, blockSize)),
                                sycl::range<1>(blockSize)
                        ),
                        [=](sycl::nd_item<1> item) {
                            const auto idx = item.get_global_id(0);
                            if (idx < capturedN) {
                                const float xi = accTnInput[idx];
                                float cube = 0.044715f * xi * xi * xi;
                                accTnOutput[idx] = 0.5f * xi * (
                                        1.0f + sycl::tanh<float>(capturedGeluScalingFactor * (xi + cube))
                                );
                            }
                        });
            });
            report();
            return event;
        }

    private:
        core::Tensor<float> &tnOutput;
        size_t outputOffset;
        const core::Tensor<float> &tnInput;
        size_t inputOffset;
        const int N;
        const float geluScalingFactor;
    };


}

