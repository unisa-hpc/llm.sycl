//
// Created by saleh on 23/05/24.
//

#pragma once

#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

#include "core/Tensor.h"
#include "BaseKernel.h"

#include "Permute.h"
#include "Unpermute.h"
#include "Softmax.h"

namespace llmsycl::kernels {

    class Attention : public BaseKernel {
        friend class sycl::handler;

    public:
        Attention(
                core::Tensor<float> &tnOut,
                size_t outOffset,
                core::Tensor<float> &tnQkvr,
                size_t qkvrOffset,
                core::Tensor<float> &tnAtt,
                size_t attOffset,
                core::Tensor<float> &tnInp,
                size_t inpOffset,
                int B, int T, int C, int NH,
                int blockSizeSoftMax
        ) :
                BaseKernel("Attention"),
                tnOut(tnOut),
                outOffset(outOffset),
                tnQkvr(tnQkvr),
                qkvrOffset(qkvrOffset),
                tnAtt(tnAtt),
                attOffset(attOffset),
                tnInp(tnInp),
                inpOffset(inpOffset),
                B(B), T(T), C(C), NH(NH), blockSizeSoftMax(blockSizeSoftMax) {

            addTensorDetailsToReport("tnOut", tnOut);
            addTensorDetailsToReport("tnQkvr", tnQkvr);
            addTensorDetailsToReport("tnAtt", tnAtt);
            addTensorDetailsToReport("tnInp", tnInp);
            addScalarParamToReport("B", B);
            addScalarParamToReport("T", T);
            addScalarParamToReport("C", C);
            addScalarParamToReport("NH", NH);
            addScalarParamToReport("blockSizeSoftMax", blockSizeSoftMax);

        }

        sycl::event Launch(
                sycl::queue &q,
                int blockSize) override {
            auto event = q.submit([&](sycl::handler &h) {
                // inp is (B, T, 3C) QKV
                // preatt, att are (B, NH, T, T)
                // output is (B, T, C)
                int HS = C / NH; // head size

                // STEP 1
                // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
                Permute permute_kernel(
                        tnQkvr, qkvrOffset,
                        tnQkvr, qkvrOffset + 1 * B * T * C,
                        tnQkvr, qkvrOffset + 2 * B * T * C,
                        tnInp, inpOffset,
                        B, T, NH, HS);
                permute_kernel.Launch(q, blockSize);


                // STEP 2 - Creating sub buffers needed for oneMKL.
                {
                    auto subBufQ = sycl::buffer(
                            tnQkvr.getDeviceBuff(),
                            sycl::id<1>(qkvrOffset + 0),
                            sycl::range<1>(B * NH * T * HS)
                    );
                    auto subBufK = sycl::buffer(
                            tnQkvr.getDeviceBuff(),
                            sycl::id<1>(qkvrOffset + 1 * B * NH * T * HS),
                            sycl::range<1>(B * NH * T * HS)
                    );
                    auto subBufV = sycl::buffer(
                            tnQkvr.getDeviceBuff(),
                            sycl::id<1>(qkvrOffset + 2 * B * NH * T * HS),
                            sycl::range<1>(B * NH * T * HS)
                    );
                    const float alpha = 1.0f;
                    const float beta = 0.0f;
                    oneapi::mkl::blas::column_major::gemm_batch(
                            q,
                            oneapi::mkl::transpose::trans,
                            oneapi::mkl::transpose::nontrans,
                            T, T, HS,
                            alpha,
                            subBufK,
                            HS, T * HS,
                            subBufQ,
                            HS, T * HS,
                            beta,
                            tnInp.getDeviceBuff(),
                            T, T * T,
                            B * NH
                    );
                }

                // STEP 3 - Softmax
                Softmax softmax_kernel(
                        tnAtt, attOffset,
                        tnInp, inpOffset,
                        1.0f / sqrtf((float) HS),
                        B * NH, T
                );
                softmax_kernel.Launch(q, blockSizeSoftMax);

                // STEP 4 - gemm
                {
                    auto subBufQ = sycl::buffer(
                            tnQkvr.getDeviceBuff(),
                            sycl::id<1>(qkvrOffset + 0),
                            sycl::range<1>(B * NH * T * HS)
                    );
                    auto subBufK = sycl::buffer(
                            tnQkvr.getDeviceBuff(),
                            sycl::id<1>(qkvrOffset + 1 * B * NH * T * HS),
                            sycl::range<1>(B * NH * T * HS)
                    );
                    auto subBufV = sycl::buffer(
                            tnQkvr.getDeviceBuff(),
                            sycl::id<1>(qkvrOffset + 2 * B * NH * T * HS),
                            sycl::range<1>(B * NH * T * HS)
                    );
                    const float alpha = 1.0f;
                    const float beta = 0.0f;
                    oneapi::mkl::blas::column_major::gemm_batch(
                            q,
                            oneapi::mkl::transpose::nontrans,
                            oneapi::mkl::transpose::nontrans,
                            HS, T, T,
                            alpha,
                            subBufV,
                            HS, T * HS,
                            tnAtt.getDeviceBuff(),
                            T, T * T,
                            beta,
                            tnInp.getDeviceBuff(),
                            HS, T * HS,
                            B * NH
                    );
                }


                // STEP 5 - Un-permute
                //(const float* inp, float *out, int B, int N, int NH, int d) {
                Unpermute unpermute_kernel(
                        tnOut, outOffset,
                        tnInp, inpOffset,
                        B, T, NH, HS,
                        Helpers::CeilDiv(B * T * C, blockSize)
                );
                unpermute_kernel.Launch(q, blockSize);

            });
            report();
            return event;
        }

    private:
        core::Tensor<float> &tnOut;
        const size_t outOffset;
        core::Tensor<float> &tnQkvr;
        const size_t qkvrOffset;
        core::Tensor<float> &tnAtt;
        const size_t attOffset;
        core::Tensor<float> &tnInp;
        const size_t inpOffset;

        const int B, T, C, NH, blockSizeSoftMax;
    };


}

