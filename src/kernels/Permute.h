//
// Created by saleh on 23/05/24.
//

#pragma once

#include <sycl/sycl.hpp>

#include "core/Tensor.h"
#include "BaseKernel.h"

namespace llmsycl::kernels {

    class Permute : public BaseKernel {
        friend class sycl::handler;

    public:
        Permute(
                core::Tensor<float> &tnQ,
                size_t qOffset,
                core::Tensor<float> &tnK,
                size_t kOffset,
                core::Tensor<float> &tnV,
                size_t vOffset,
                const core::Tensor<float> &tnInp,
                size_t inpOffset,
                int B, int N, int NH, int d
        ) :
                BaseKernel("Permute"),
                tnQ(tnQ), qOffset(qOffset),
                tnK(tnK), kOffset(kOffset),
                tnV(tnV), vOffset(vOffset),
                tnInp(tnInp), inpOffset(inpOffset),
                B(B), N(N), NH(NH), d(d) {

            addTensorDetailsToReport("tnQ", tnQ);
            addTensorDetailsToReport("tnK", tnK);
            addTensorDetailsToReport("tnV", tnV);
            addTensorDetailsToReport("tnInp", tnInp);
            addScalarParamToReport("B", B);
            addScalarParamToReport("N", N);
            addScalarParamToReport("NH", NH);
            addScalarParamToReport("d", d);
        }

        sycl::event Launch(sycl::queue &q, int blockSize) override {
            auto event = q.submit([&](sycl::handler &h) {
                auto accTnQ = tnQ.getAccessorDeviceWrite(h, qOffset);
                auto accTnK = tnK.getAccessorDeviceWrite(h, kOffset);
                auto accTnV = tnV.getAccessorDeviceWrite(h, vOffset);
                auto accTnInp = tnInp.getAccessorDeviceRead(h, inpOffset);

                const int capturedB = this->B;
                const int capturedN = this->N;
                const int capturedNH = this->NH;
                const int capturedD = this->d;
                const int upperBound = B*NH*N*d;

                h.parallel_for(
                        sycl::nd_range<1>(
                                sycl::range<1>(Helpers::MakeDivisible(upperBound, blockSize)),
                                sycl::range<1>(blockSize)
                        ),
                        [=](sycl::nd_item<1> item) {
                            const int idx = (int) item.get_global_id(0);
                            if (idx < upperBound) {
                                // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
                                // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
                                // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]

                                int b = idx / (capturedNH * capturedN * capturedD);
                                int rest = idx % (capturedNH * capturedN * capturedD);
                                int nh_ = rest / (capturedN * capturedD);
                                rest = rest % (capturedN * capturedD);
                                int n = rest / capturedD;
                                int d_ = rest % capturedD;
                                int inp_idx =
                                        (b * capturedN * 3 * capturedNH * capturedD) +
                                        (n * 3 * capturedNH * capturedD) +
                                        (0 * capturedNH * capturedD) +
                                        (nh_ * capturedD) +
                                        d_;
                                accTnQ[idx] = accTnInp[inp_idx];
                                accTnK[idx] = accTnInp[inp_idx + capturedNH * capturedD];
                                accTnV[idx] = accTnInp[inp_idx + 2 * (capturedNH * capturedD)];
                            }
                        });
            });
            report();
            return event;
        }

    private:
        core::Tensor<float> &tnQ;
        size_t qOffset;
        core::Tensor<float> &tnK;
        size_t kOffset;
        core::Tensor<float> &tnV;
        size_t vOffset;
        const core::Tensor<float> &tnInp;
        size_t inpOffset;
        const int B, N, NH, d;
    };


}

