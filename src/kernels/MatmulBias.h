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
                core::Tensor<float> &tnOut,
                size_t outOffset,
                core::Tensor<float> &tnInp,
                size_t inpOffset,
                core::Tensor<float> &tnWeight,
                size_t weightOffset,
                core::Tensor<float> &tnBias,
                size_t biasOffset,
                int B, int T, int C, int OC,
                bool hasBias = true
        ) :
                BaseKernel("Matmul"),
                tnOut(tnOut), outOffset(outOffset),
                tnInp(tnInp), inpOffset(inpOffset),
                tnWeight(tnWeight), weightOffset(weightOffset),
                tnBias(tnBias), biasOffset(biasOffset),
                B(B), T(T), C(C), OC(OC), hasBias(hasBias) {

            addTensorDetailsToReport("tnOut", tnOut);
            addTensorDetailsToReport("tnInp", tnInp);
            addTensorDetailsToReport("tnWeight", tnWeight);
            addTensorDetailsToReport("tnBias", tnBias);

            addScalarParamToReport("B", B);
            addScalarParamToReport("T", T);
            addScalarParamToReport("C", C);
            addScalarParamToReport("OC", OC);
        }

        sycl::event Launch(sycl::queue &q, int blockSize) override {
            sycl::event event;
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
                auto subBufW = sycl::buffer(
                        tnWeight.getDeviceBuff(),
                        sycl::id<1>(weightOffset),
                        sycl::range<1>(C * OC)
                );
                auto subBufI = sycl::buffer(
                        tnInp.getDeviceBuff(),
                        sycl::id<1>(inpOffset),
                        sycl::range<1>(B * T * C)
                );
                auto subBufO = sycl::buffer(
                        tnOut.getDeviceBuff(),
                        sycl::id<1>(outOffset),
                        sycl::range<1>(B * T * OC)
                );
                const float alpha = 1.0f;
                const float beta = 0.0f;
                //cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, B*T, C, &alpha, weight, C, inp, C, &beta, out, OC));
                oneapi::mkl::blas::column_major::gemm(
                        q,
                        oneapi::mkl::transpose::trans,
                        oneapi::mkl::transpose::nontrans,
                        OC, B * T, C,  //m, n, k
                        alpha,
                        subBufW,
                        C,
                        subBufI,
                        C,
                        beta,
                        subBufO,
                        OC
                );
            }

            if (hasBias) {
                event = q.submit([&](sycl::handler &h) {
                    auto accTnOut = tnOut.getAccessorDeviceWrite(h, outOffset);
                    auto accTnInp = tnInp.getAccessorDeviceRead(h, inpOffset);
                    auto accTnWeight = tnWeight.getAccessorDeviceRead(h, weightOffset);
                    auto accTnBias = tnBias.getAccessorDeviceRead(h, biasOffset);

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
                                    accTnOut[b * capturedT * capturedOC + t * capturedOC + oc] += accTnBias[oc];
                                }
                            }
                    );

                });
            }
            report();
            return event;
        }

    private:
        core::Tensor<float> &tnOut;
        size_t outOffset;
        core::Tensor<float> &tnInp;
        size_t inpOffset;
        core::Tensor<float> &tnWeight;
        size_t weightOffset;
        core::Tensor<float> &tnBias;
        size_t biasOffset;
        const int B, T, C, OC;
        const bool hasBias;
    };


}

