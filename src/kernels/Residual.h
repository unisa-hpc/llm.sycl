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
                core::Tensor<float> &tnOutput,
                size_t outputOffset,
                core::Tensor<float> &tnInput1,
                size_t input1Offset,
                core::Tensor<float> &tnInput2,
                size_t input2Offset,
                int N ) :
                BaseKernel("Residual"),
                tnOutput(tnOutput), outputOffset(outputOffset),
                tnInput1(tnInput1), input1Offset(input1Offset),
                tnInput2(tnInput2), input2Offset(input2Offset),
                N(N)  {

            addTensorDetailsToReport("tnOutput", tnOutput);
            addTensorDetailsToReport("tnInput1", tnInput1);
            addTensorDetailsToReport("tnInput2", tnInput2);
            addScalarParamToReport("outputOffset", outputOffset);
            addScalarParamToReport("input1Offset", input1Offset);
            addScalarParamToReport("input1Offset", input2Offset);
            addScalarParamToReport("N", N);
        }

        sycl::event Launch(sycl::queue &q, int blockSize) override {
            auto event = q.submit([&](sycl::handler &h) {
                auto accTnOutput = tnOutput.getAccessorDeviceWrite(h, outputOffset);
                auto accTnInput1 = tnInput1.getAccessorDeviceWrite(h, input1Offset);
                auto accTnInput2 = tnInput2.getAccessorDeviceWrite(h, input2Offset);
                const size_t capturedN = this->N;

                h.parallel_for(
                        sycl::nd_range<1>(
                                sycl::range<1>(Helpers::MakeDivisible(N, blockSize)),
                                sycl::range<1>(blockSize)
                        ),
                        [=](sycl::nd_item<1> item) {
                            const auto idx = item.get_global_id(0);
                            if (idx < capturedN) {
                                accTnOutput[idx] = accTnInput1[idx] + accTnInput2[idx];
                            }
                        });
            });
            report();
            return event;
        }

    private:
        core::Tensor<float> &tnOutput;
        size_t outputOffset;
        core::Tensor<float> &tnInput1;
        size_t input1Offset;
        core::Tensor<float> &tnInput2;
        size_t input2Offset;
        const int N;
    };


}

