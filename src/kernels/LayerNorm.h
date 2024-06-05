//
// Created by saleh on 23/05/24.
//

#pragma once

#include <sycl/sycl.hpp>

#include "core/Tensor.h"
#include "BaseKernel.h"

namespace llmsycl::kernels {

    class LayerNorm : public BaseKernel {
        friend class sycl::handler;

    public:
        LayerNorm(
                float *dOut,
                float *dMean,
                float *dRstd,
                const float *dInp,
                const float *dWeight,
                const float *dBias,
                int B, int T, int C
        ) :
                BaseKernel("LayerNorm"),
                dOut(dOut),
                dMean(dMean),
                dRstd(dRstd),
                dInp(dInp),
                dWeight(dWeight),
                dBias(dBias),
                B(B), T(T), C(C) {
            addScalarParamToReport("B", B);
            addScalarParamToReport("T", T);
            addScalarParamToReport("C", C);
        }

        std::vector<sycl::event> Launch(sycl::queue &q, int blockSize) override {
            auto event = q.submit([&](sycl::handler &h) {
                auto capturedOut = dOut;
                auto capturedMean = dMean;
                auto capturedRstd = dRstd;
                auto capturedInp = dInp;
                auto capturedWeight = dWeight;
                auto capturedBias = dBias;

                const int capturedN = this->B * this->T;
                const int capturedC = this->C;

                h.parallel_for(
                        sycl::nd_range<1>(
                                sycl::range<1>(Helpers::MakeDivisible(B * T, blockSize)),
                                sycl::range<1>(blockSize)
                        ),
                        [=](sycl::nd_item<1> item) {
                            const int idx = (int) item.get_global_id(0);
                            if (idx < capturedN) {
                                float eps = 1e-5f;

                                // calculate the mean
                                float m = 0.0f;
                                for (int i = 0; i < capturedC; i++) {
                                    m += capturedInp[idx * capturedC + i];
                                }
                                m = m / (float)capturedC;
                                // calculate the variance (without any bias correction)
                                float v = 0.0f;
                                for (int i = 0; i < capturedC; i++) {
                                    float xshift = capturedInp[idx * capturedC + i] - m;
                                    v += xshift * xshift;
                                }
                                v = v / (float)capturedC;
                                // calculate the rstd
                                float s = 1.0f / sycl::sqrt(v + eps);


                                for (int i = 0; i < capturedC; i++) {
                                    float n = (s * (capturedInp[idx * capturedC + i] - m)); // normalized output
                                    float o = n * capturedWeight[i] + capturedBias[i]; // scale and shift it
                                    capturedOut[idx * capturedC + i] = o; // write
                                }
                                // cache the mean and rstd for the backward pass later
                                capturedMean[idx] = m;
                                capturedRstd[idx] = s;
                            }
                        });
            });
            report();
            return {event};
        }

    private:
        float *dOut;
        float *dMean;
        float *dRstd;
        const float *dInp;
        const float *dWeight;
        const float *dBias;
        const int B, T, C;
    };


}

