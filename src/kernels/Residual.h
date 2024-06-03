//
// Created by saleh on 23/05/24.
//

#pragma once

#include <sycl/sycl.hpp>

#include "core/Tensor.h"
#include "BaseKernel.h"

namespace llmsycl::kernels {

    class Residual : public BaseKernel {
        friend class sycl::handler;

    public:
        Residual(
                float *dOutput,
                const float *dInput1,
                const float *dInput2,
                int N ) :
                BaseKernel("Residual"),
                dOutput(dOutput),
                dInput1(dInput1),
                dInput2(dInput2),
                N(N)  {

            addScalarParamToReport("N", N);
        }

        std::vector<sycl::event> Launch(sycl::queue &q, int blockSize) override {
            auto event = q.submit([&](sycl::handler &h) {
                auto capturedOutput = dOutput;
                auto capturedInput1 = dInput1;
                auto capturedInput2 = dInput2;
                const size_t capturedN = this->N;

                h.parallel_for(
                        sycl::nd_range<1>(
                                sycl::range<1>(Helpers::MakeDivisible(N, blockSize)),
                                sycl::range<1>(blockSize)
                        ),
                        [=](sycl::nd_item<1> item) {
                            const auto idx = item.get_global_id(0);
                            if (idx < capturedN) {
                                capturedOutput[idx] = capturedInput1[idx] + capturedInput2[idx];
                            }
                        });
            });
            report();
            return {event};
        }

    private:
        float *dOutput;
        const float *dInput1;
        const float *dInput2;
        const int N;
    };


}

