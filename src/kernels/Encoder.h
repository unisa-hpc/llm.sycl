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
                float *dOut,
                const int *dIn,
                const float *dWte,
                const float *dWpe,
                int B, int T, int C) :
                BaseKernel("EncoderKernel"),
                dOut(dOut),
                dIn(dIn),
                dWte(dWte),
                dWpe(dWpe),
                B(B), T(T), C(C) {

            addScalarParamToReport("B", B);
            addScalarParamToReport("T", T);
            addScalarParamToReport("C", C);
        }

        std::vector<sycl::event> Launch(
                sycl::queue &q,
                int blockSize,
                const std::vector<sycl::event> &dependencies) override {

            auto event = q.submit([&](sycl::handler &h) {

                h.depends_on(dependencies);

                const int bound = this->B * this->T * this->C;
                const int capturedB = this->B;
                const int capturedT = this->T;
                const int capturedC = this->C;
                auto capturedWte = this->dWte;
                auto capturedWpe = this->dWpe;
                auto capturedOut = this->dOut;
                auto capturedIn = this->dIn;
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
                             * wpe: maxT x C
                             */
                            if (idx < bound) {
                                const int bt = idx / capturedC;
                                const int b = bt / capturedT;
                                const int t = bt % capturedT;
                                const int c = idx % capturedC;
                                const int ix = capturedIn[b * capturedT + t];
                                //out[b * T * C4 + t * C4 + c4] = add_float4(wte[ix * C4 + c4], wpe[t * C4 + c4]);
                                capturedOut[b * capturedT * capturedC + t * capturedC + c] =
                                        capturedWte[ix * capturedC + c] + capturedWpe[t * capturedC + c];
                            }
                        });
            });
            report();
            return {event};
        }

    private:
        float *dOut;
        const int *dIn;
        const float *dWte;
        const float *dWpe;
        const int B, T, C;
    };


}

