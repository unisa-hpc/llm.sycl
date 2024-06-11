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

        std::vector<sycl::event> Launch(
                sycl::queue &q,
                int blockSize,
                const std::vector<sycl::event> &dependencies) override {

            auto event = q.submit([&](sycl::handler &h) {

                h.depends_on(dependencies);

                auto capturedOut = dOut;
                auto capturedMean = dMean;
                auto capturedRstd = dRstd;
                auto capturedInp = dInp;
                auto capturedWeight = dWeight;
                auto capturedBias = dBias;

                const int capturedN = this->B * this->T;
                const int capturedC = this->C;
                if (capturedC%4 != 0) {
                    throw std::runtime_error("LayerNorm: C must be divisible by 4");
                }

                h.parallel_for(
                        sycl::nd_range<1>(
                                sycl::range<1>(Helpers::MakeDivisible(B * T, blockSize)),
                                sycl::range<1>(blockSize)
                        ),
                        /*[=](sycl::nd_item<1> item) {
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
                         */
                        [=](sycl::nd_item<1> item) {
                            const int idx = (int) item.get_global_id(0);
                            if (idx < capturedN) {
                                sycl::float4 eps = {1e-5f, 1e-5f, 1e-5f, 1e-5f};

                                // calculate the mean
                                sycl::float4 mVec = {0.0f, 0.0f, 0.0f, 0.0f};
                                for (int i = 0; i < capturedC/4; i++) {
                                    mVec += ((sycl::float4 *)capturedInp)[idx * capturedC/4 + i];
                                }
                                auto m = (mVec[0] + mVec[1] + mVec[2] + mVec[3]) / (float)capturedC;

                                // calculate the variance (without any bias correction)
                                sycl::float4 vVec = {0.0f, 0.0f, 0.0f, 0.0f};
                                for (int i = 0; i < capturedC/4; i++) {
                                    sycl::float4 xshift = ((sycl::float4 *)capturedInp)[idx * capturedC/4 + i] - m;
                                    vVec += xshift * xshift;
                                }
                                auto v = (vVec[0] + vVec[1] + vVec[2] + vVec[3]) / (float)capturedC;
                                // calculate the rstd
                                float s = 1.0f / sycl::sqrt(v + eps[0]);

                                for (int i = 0; i < capturedC/4; i++) {
                                    sycl::float4 n = (s * (((sycl::float4 *)capturedInp)[idx * capturedC/4 + i] - m)); // normalized output
                                    sycl::float4 o = n * ((sycl::float4 *)capturedWeight)[i] + ((sycl::float4 *)capturedBias)[i]; // scale and shift it
                                    ((sycl::float4 *)capturedOut)[idx * capturedC/4 + i] = o; // write
                                }
                                // cache the mean and rstd for the backward pass later
                                // Note: This part needs to be handled separately as it's not clear how to vectorize it
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

