//
// Created by saleh on 23/05/24.
//

#pragma once

#include <sycl/sycl.hpp>

#include "core/Tensor.h"
#include "BaseKernel.h"
#include <oneapi/mkl.hpp>

namespace llmsycl::kernels {

    class MatmulBias : public BaseKernel {
        friend class sycl::handler;

    public:
        MatmulBias(
                float *dOut,
                const float *dInp,
                const float *dWeight,
                const float *dBias,
                int B, int T, int C, int OC,
                bool hasBias = true
        ) :
                BaseKernel("Matmul"),
                dOut(dOut),
                dInp(dInp),
                dWeight(dWeight),
                dBias(dBias),
                B(B), T(T), C(C), OC(OC), hasBias(hasBias) {

            addScalarParamToReport("B", B);
            addScalarParamToReport("T", T);
            addScalarParamToReport("C", C);
            addScalarParamToReport("OC", OC);
            addScalarParamToReport("hasBias", hasBias);
        }

        std::vector<sycl::event> Launch(sycl::queue &q, int blockSize) override {
            std::vector<sycl::event> events;
            auto capturedInp = dInp;
            auto capturedWeight = dWeight;
            auto capturedBias = dBias;
            auto capturedOut = dOut;

            {
                // inp is (B*T, C)  : transpose (col-mjr) : k=C, n=B*T
                // weight is (OC, C) : transpose : transpose (col-mjr) : m=OC, k=C
                // out is (B*T, OC)

                /**
                 * We need "weight.T @ inp"
                 * Gemm does:
                C := alpha*op(A)*op(B) + beta*C
                 op(X) is one of op(X) = X or op(X) = X', or op(X) = conjg(X')
                 alpha and beta are scalars
                 A, B and C are matrices:
                    A is an m by k matrix
                    B is a k by n matrix
                    C is an m by n matrix
                */

                const float alpha = 1.0f;
                const float beta = 0.0f;
                //cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, B*T, C, &alpha, weight, C, inp, C, &beta, out, OC));
                auto event = oneapi::mkl::blas::column_major::gemm(
                        q,
                        oneapi::mkl::transpose::trans,
                        oneapi::mkl::transpose::nontrans,
                        OC, B * T, C,  //m, n, k
                        alpha,
                        capturedWeight,
                        C,
                        capturedInp,
                        C,
                        beta,
                        dOut,
                        OC
                );
                events.push_back(event);
            }

            if (hasBias) {
                auto event = q.submit([&](sycl::handler &h) {
                    const int capturedB = this->B;
                    const int capturedT = this->T;
                    const int capturedC = this->C;
                    const int capturedOC = this->OC;

                    /*
                    int block_size = sqrt_block_size * sqrt_block_size;
                    int grid_size = ceil_div(OC * B * T, block_size);
                    add_bias<<<grid_size, block_size>>>(out, bias, B, T, OC);
                    */
                    h.parallel_for(
                            sycl::nd_range<1>(
                                    sycl::range<1>(Helpers::MakeDivisible(OC * B * T, blockSize)),
                                    sycl::range<1>(blockSize)
                            ),
                            [=](sycl::nd_item<1> item) {
                                auto idx = item.get_global_id(0);
                                if (idx < capturedOC * capturedB * capturedT) {
                                    int b = idx / (capturedT * capturedOC);
                                    int t = (idx % (capturedT * capturedOC)) / capturedOC;
                                    int oc = idx % capturedOC;
                                    capturedOut[b * capturedT * capturedOC + t * capturedOC + oc] += capturedBias[oc];
                                }
                            }
                    );

                });
                events.push_back(event);
            }
            report();
            return events;
        }

    private:
        float *dOut;
        const float *dInp;
        const float *dWeight;
        const float *dBias;

        const int B, T, C, OC;
        const bool hasBias;
    };


}

