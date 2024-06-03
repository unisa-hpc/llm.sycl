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
                float *dQ,
                float *dK,
                float *dV,
                const float *dInp,
                int B, int N, int NH, int d
        ) :
                BaseKernel("Permute"),
                dQ(dQ),
                dK(dK),
                dV(dV),
                dInp(dInp),
                B(B), N(N), NH(NH), d(d) {

            addScalarParamToReport("B", B);
            addScalarParamToReport("N", N);
            addScalarParamToReport("NH", NH);
            addScalarParamToReport("d", d);
        }

        std::vector<sycl::event> Launch(sycl::queue &q, int blockSize) override {
            auto event = q.submit([&](sycl::handler &h) {
                auto capturedQ = dQ;
                auto capturedK = dK;
                auto capturedV = dV;
                auto capturedInp = dInp;

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
                                capturedQ[idx] = capturedInp[inp_idx];
                                capturedK[idx] = capturedInp[inp_idx + capturedNH * capturedD];
                                capturedV[idx] = capturedInp[inp_idx + 2 * (capturedNH * capturedD)];
                            }
                        });
            });
            report();
            return {event};
        }

    private:
        float *dQ;
        float *dK;
        float *dV;
        const float *dInp;
        const int B, N, NH, d;
    };


}

