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
                float *dOutput,
                const float *dInput,
                int N) :
                BaseKernel("Gelu"),
                dOutput(dOutput),
                dInput(dInput),
                N(N), geluScalingFactor(std::sqrt(2.0f / M_PI)) {

            addScalarParamToReport("N", N);
            addScalarParamToReport("geluScalingFactor", geluScalingFactor);
        }

        std::vector<sycl::event> Launch(sycl::queue &q, int blockSize) override {
            auto event = q.submit([&](sycl::handler &h) {
                auto capturedOutput = dOutput;
                auto capturedInput = dInput;
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
                                const float xi = capturedInput[idx];
                                float cube = 0.044715f * xi * xi * xi;
                                capturedOutput[idx] = 0.5f * xi * (
                                        1.0f + sycl::tanh<float>(capturedGeluScalingFactor * (xi + cube))
                                );
                            }
                        });
            });
            report();
            return {event};
        }

    private:
        float *dOutput;
        const float *dInput;
        const int N;
        const float geluScalingFactor;
    };


}

