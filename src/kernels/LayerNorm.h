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
                core::Tensor<float> &tnOut,
                size_t outOffset,
                core::Tensor<float> &tnMean,
                size_t meanOffset,
                core::Tensor<float> &tnRstd,
                size_t rstdOffset,
                core::Tensor<float> &tnInp,
                size_t inpOffset,
                core::Tensor<float> &tnWeight,
                size_t weightOffset,
                core::Tensor<float> &tnBias,
                size_t biasOffset,
                int B, int T, int C
        ) :
                BaseKernel("LayerNorm"),
                tnOut(tnOut),
                outOffset(outOffset),
                tnMean(tnMean),
                meanOffset(meanOffset),
                tnRstd(tnRstd),
                rstdOffset(rstdOffset),
                tnInp(tnInp),
                inpOffset(inpOffset),
                tnWeight(tnWeight),
                weightOffset(weightOffset),
                tnBias(tnBias),
                biasOffset(biasOffset),
                B(B), T(T), C(C) {
            addTensorDetailsToReport("tnOut", tnOut);
            addTensorDetailsToReport("tnMean", tnMean);
            addTensorDetailsToReport("tnRstd", tnRstd);
            addTensorDetailsToReport("tnInp", tnInp);
            addTensorDetailsToReport("tnWeight", tnWeight);
            addTensorDetailsToReport("tnBias", tnBias);
            addScalarParamToReport("B", B);
            addScalarParamToReport("T", T);
            addScalarParamToReport("C", C);
        }

        sycl::event Launch(sycl::queue &q, int blockSize) override {
            auto event = q.submit([&](sycl::handler &h) {

                auto accTnOut = tnOut.getAccessorDeviceWrite(h, outOffset);
                auto accTnMean = tnMean.getAccessorDeviceWrite(h, meanOffset);
                auto accTnRstd = tnRstd.getAccessorDeviceWrite(h, rstdOffset);
                auto accTnInp = tnInp.getAccessorDeviceRead(h, inpOffset);
                auto accTnWeight = tnWeight.getAccessorDeviceRead(h, weightOffset);
                auto accTnBias = tnBias.getAccessorDeviceRead(h, biasOffset);

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
                                    m += accTnInp[idx * capturedC + i];
                                }
                                m = m / capturedC;
                                // calculate the variance (without any bias correction)
                                float v = 0.0f;
                                for (int i = 0; i < capturedC; i++) {
                                    float xshift = accTnInp[idx * capturedC + i] - m;
                                    v += xshift * xshift;
                                }
                                v = v / capturedC;
                                // calculate the rstd
                                float s = 1.0f / sqrtf(v + eps);


                                for (int i = 0; i < capturedC; i++) {
                                    float n = (s * (accTnInp[idx * capturedC + i] - m)); // normalized output
                                    float o = n * accTnWeight[i] + accTnBias[i]; // scale and shift it
                                    accTnOut[idx * capturedC + i] = o; // write
                                }
                                // cache the mean and rstd for the backward pass later
                                accTnMean[idx] = m;
                                accTnRstd[idx] = s;
                            }
                        });
            });
            report();
            return event;
        }

    private:
        core::Tensor<float> &tnOut;
        const size_t outOffset;
        core::Tensor<float> &tnMean;
        const size_t meanOffset;
        core::Tensor<float> &tnRstd;
        const size_t rstdOffset;
        core::Tensor<float> &tnInp;
        const size_t inpOffset;
        core::Tensor<float> &tnWeight;
        const size_t weightOffset;
        core::Tensor<float> &tnBias;
        const size_t biasOffset;
        const int B, T, C;
    };


}

