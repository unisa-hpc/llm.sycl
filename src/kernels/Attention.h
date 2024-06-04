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
                float *dOut,
                float *dQkvr,
                float *dAtt,
                float *dInp,
                int B, int T, int C, int NH,
                int blockSizeSoftMax
        ) :
                BaseKernel("Attention"),
                dOut(dOut),
                dQkvr(dQkvr),
                dAtt(dAtt),
                dInp(dInp),
                B(B), T(T), C(C), NH(NH), blockSizeSoftMax(blockSizeSoftMax) {

            addScalarParamToReport("B", B);
            addScalarParamToReport("T", T);
            addScalarParamToReport("C", C);
            addScalarParamToReport("NH", NH);
            addScalarParamToReport("blockSizeSoftMax", blockSizeSoftMax);

        }

        /**
         * @details
         *    input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
         *    preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
         *    that holds the pre-attention and post-attention scores (used in backward)
         *    output is (B, T, C)
         *    attention is the only layer that mixes information across time
         *    every other operation is applied at every (b,t) position independently
         *    (and of course, no layer mixes information across batch)
         *
         *    inp is (B, T, 3C) QKV
         *    preatt, att are (B, NH, T, T)
         *    output is (B, T, C)
         *
         * @note Have a look at these possible values:
         *    B: 1                                    : Batch size
         *    T: 1024                                 : Sequence Length
         *    C: 768                                  : ?
         *    NH: 12                                  : Number of heads
         *    HS: C / NH = 64                         : Head size
         *    tnQkvr: B*T*3*C
         *    tnQ, tnK, tnV: B*T*C == B*T*NH*HS       : Query, Key, Value
         *    tnAtt: B*NH*T*T                         : Attention
         *    tnOut: B*T*NH*HS                        : ?
         *    tnInp: B*T*3*C                          : Mangled Query, Key, Value / SCRATCH PAD AFTERWARD
         * @param q
         * @param blockSize
         * @return
         */
        std::vector<sycl::event> Launch(
                sycl::queue &q,
                int blockSize) override {

            std::vector<sycl::event> events;
            int HS = C / NH; // head size

            // STEP 1
            // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
            {
                /// TODO: tnInp.save(inpOffset + 0, B * T * 3 * C, "/tmp/xInp_uut.npy");
                Permute permute_kernel(
                        dQkvr + 0 * B * T * C,
                        dQkvr + 1 * B * T * C,
                        dQkvr + 2 * B * T * C,
                        dInp,
                        B, T, NH, HS);

                auto e = permute_kernel.Launch(q, blockSize);
                for (auto &event: e) {
                    events.push_back(event);
                }
            }
            /// TODO: tnQkvr.save(qkvrOffset + 0 * B * T * C, B * T * C, "/tmp/xq_uut.npy"); // ok
            /// TODO: tnQkvr.save(qkvrOffset + 1 * B * T * C, B * T * C, "/tmp/xk_uut.npy"); // ok
            /// TODO: tnQkvr.save(qkvrOffset + 2 * B * T * C, B * T * C, "/tmp/xv_uut.npy"); // ok

            auto dQuery = dQkvr + 0 * B * T * C;
            auto dKey = dQkvr + 1 * B * T * C;
            auto dValue = dQkvr + 2 * B * T * C;
            auto preAtt = dInp;

            // STEP 2 - Creating sub buffers needed for oneMKL.
            //core::Tensor<float> tnTmp({(size_t) B * NH * T * T});
            {
                sycl::buffer<float, 1> dCopyK(sycl::range<1>(B * NH * T * HS));
                sycl::buffer<float, 1> dCopyQ(sycl::range<1>(B * NH * T * HS));
                const float alpha = 1.0f;
                const float beta = 0.0f;

                auto e = oneapi::mkl::blas::column_major::gemm_batch(
                        q,
                        oneapi::mkl::transpose::trans,
                        oneapi::mkl::transpose::nontrans,
                        T, T, HS,
                        alpha,
                        dKey,
                        HS, T * HS,
                        dQuery,
                        HS, T * HS,
                        beta,
                        preAtt,
                        T, T * T,
                        B * NH
                );
                events.push_back(e);
            }


            // STEP 3 - Softmax
            {
                //tnTmp.save( "/tmp/xa_uut.npy"); // Almost ok
                ///TODO: tnInp.save(0, B * NH * T * T, "/tmp/xa_uut.npy"); // WRONG
                Softmax softmax_kernel(
                        dAtt,
                        preAtt,
                        1.0f / std::sqrt((float) HS),
                        B * NH * T, T
                );
                auto e = softmax_kernel.Launch(q, blockSizeSoftMax);
                for (auto &event: e) {
                    events.push_back(event);
                }
                ///TODO: tnAtt.save(attOffset + 0, B * NH * T * T, "/tmp/xb_uut.npy"); // WRONG
            }


            // STEP 4 - gemm
            {
                sycl::buffer<float, 1> dCopyV(sycl::range<1>(B * NH * T * HS));
                sycl::buffer<float, 1> dCopyAtt(sycl::range<1>(B * NH * T * T));

                const float alpha = 1.0f;
                const float beta = 0.0f;
                // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
                oneapi::mkl::blas::column_major::gemm_batch(
                        q,
                        oneapi::mkl::transpose::nontrans,
                        oneapi::mkl::transpose::nontrans,
                        HS, T, T,
                        alpha,
                        // subBufV,
                        dValue,
                        HS, T * HS,
                        dAtt,
                        T, T * T,
                        beta,
                        dInp,
                        HS, T * HS,
                        B * NH
                );
            }


            // STEP 5 - Un-permute
            {
                Unpermute unpermute_kernel(
                        dOut,
                        dInp,
                        B, T, NH, HS,
                        Helpers::CeilDiv(B * T * C, blockSize)
                );
                auto e = unpermute_kernel.Launch(q, blockSize);
                for (auto &event: e) {
                    events.push_back(event);
                }
            }

            report();
            return events;
        }

    private:
        float *dOut;
        float *dQkvr;
        float *dAtt;
        float *dInp;

        const int B, T, C, NH, blockSizeSoftMax;
    };


}

