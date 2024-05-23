//
// Created by saleh on 23/05/24.
//

#pragma once

#include <sycl/sycl.hpp>

#include "core/Tensor.h"
#include "BaseKernel.h"

namespace llmsycl::kernels {

    class EncoderKernel : public BaseKernel {
        friend class sycl::handler;

    public:
        EncoderKernel(
                core::Tensor<float> &tnOut,
                size_t outOffset,
                const core::Tensor<int> &tnIn,
                size_t inOffset,
                const core::Tensor<float> &tnWte,
                size_t wteOffset,
                const core::Tensor<float> &tnWpe,
                size_t wpeOffset,
                int B, int T, int C) :
                BaseKernel("EncoderKernel"),
                tnOut(tnOut),
                tnIn(tnIn),
                tnWte(tnWte),
                tnWpe(tnWpe),
                B(B), T(T), C(C),
                offsetOut(outOffset),
                offsetIn(inOffset),
                offsetWte(wteOffset),
                offsetWpe(wpeOffset) {

            addTensorDetailsToReport("tnOut", tnOut);
            addTensorDetailsToReport("tnIn", tnIn);
            addTensorDetailsToReport("tnWte", tnWte);
            addTensorDetailsToReport("tnWpe", tnWpe);
            addScalarParamToReport("B", B);
            addScalarParamToReport("T", T);
            addScalarParamToReport("C", C);
        }

        void Launch(sycl::queue &q, int blockSize) override {
            q.submit([&](sycl::handler &h) {
                auto accTnOut = tnOut.getAccessorDeviceWrite(h, offsetOut);
                auto accTnIn = tnIn.getAccessorDeviceRead(h, offsetIn);
                auto accTnWte = tnWte.getAccessorDeviceRead(h, offsetWte);
                auto accTnWpe = tnWpe.getAccessorDeviceRead(h, offsetWpe);

                const int bound = this->B * this->T * this->C;
                const int capturedB = this->B;
                const int capturedT = this->T;
                const int capturedC = this->C;

                logger->trace("bound: {}", bound);
                logger->trace("capturedB: {}", capturedB);
                logger->trace("capturedT: {}", capturedT);
                logger->trace("capturedC: {}", capturedC);
                logger->trace("blockSize: {}", blockSize);
                logger->trace("offsetOut: {}", offsetOut);
                logger->trace("offsetIn: {}", offsetIn);
                logger->trace("offsetWte: {}", offsetWte);
                logger->trace("offsetWpe: {}", offsetWpe);
                logger->trace("globalSize: {}", Helpers::MakeDivisible(B * T * C, blockSize));

                h.parallel_for(
                        sycl::nd_range<1>(
                                sycl::range<1>(Helpers::MakeDivisible(B * T * C, blockSize)),
                                sycl::range<1>(blockSize)
                        ),
                        [=](sycl::nd_item<1> item) {
                            const int idx = (int) item.get_global_id(0);
                            /* out: B x T x C
                             * inp: B x T
                             * wte: V x C
                             * wpe: T x C
                             */
                            if (idx < bound) {
                                const int bt = idx / capturedC;
                                const int b = bt / capturedT;
                                const int t = bt % capturedT;
                                const int c = idx % capturedC;
                                const int ix = accTnIn[b * capturedT + t];
                                accTnOut[idx] = accTnWte[ix * capturedC + c] + accTnWpe[t * capturedC + c];
                            }
                        });
            });
            report();
        }

    private:
        core::Tensor<float> &tnOut;
        const core::Tensor<int> &tnIn;
        const core::Tensor<float> &tnWte;
        const core::Tensor<float> &tnWpe;
        const int B, T, C;
        const size_t offsetOut, offsetIn, offsetWte, offsetWpe;
    };


}

