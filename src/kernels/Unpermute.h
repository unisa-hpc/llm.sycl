//
// Created by saleh on 23/05/24.
//

#pragma once

#include <sycl/sycl.hpp>

#include "core/Tensor.h"
#include "BaseKernel.h"

namespace llmsycl::kernels {

    class Unpermute : public BaseKernel {
        friend class sycl::handler;

    public:
        Unpermute(
                float *dOut,
                const float *dInp,
                int B, int N, int NH, int d,
                int gridSize
        ) :
                BaseKernel("Unpermute"),
                dOut(dOut),
                dInp(dInp),
                B(B), N(N), NH(NH), d(d), gridSize(gridSize) {

            addScalarParamToReport("B", B);
            addScalarParamToReport("N", N);
            addScalarParamToReport("NH", NH);
            addScalarParamToReport("d", d);
        }

        std::vector<sycl::event> Launch(sycl::queue &q, int blockSize) override {
            auto event = q.submit([&](sycl::handler &h) {
                auto capturedOut = dOut;
                auto capturedInp = dInp;

                const int capturedB = this->B;
                const int capturedN = this->N;
                const int capturedNH = this->NH;
                const int capturedD = this->d;

                const int workSize = gridSize * blockSize;

                h.parallel_for(
                        sycl::nd_range<1>(
                                sycl::range<1>(workSize),
                                sycl::range<1>(blockSize)
                        ),
                        [=](sycl::nd_item<1> item) {
                            const int idx = (int) item.get_global_id(0);
                            if (idx < workSize) {
                                // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)

                                int b = idx / (capturedNH * capturedN * capturedD);
                                int rest = idx % (capturedNH * capturedN * capturedD);
                                int nh_ = rest / (capturedN * capturedD);
                                rest = rest % (capturedN * capturedD);
                                int n = rest / capturedD;
                                int d_ = rest % capturedD;
                                int other_idx = (b * capturedNH * capturedN * capturedD) + (n * capturedNH * capturedD) + (nh_ * capturedD) + d_;
                                capturedOut[other_idx] = capturedInp[idx];
                            }
                        });
            });
            report();
            return {event};
        }

    private:
        float *dOut;
        const float *dInp;
        const int B, N, NH, d, gridSize;
    };


}

