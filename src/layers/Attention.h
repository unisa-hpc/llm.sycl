//
// Created by saleh on 23/05/24.
//

#pragma once

#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

#include "core/Tensor.h"
#include "BaseLayer.h"
#include "kernels/Permute.h"
#include "kernels/Unpermute.h"
#include "kernels/Softmax.h"

namespace llmsycl::layers {

    class Attention : public BaseLayer {
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
                BaseLayer("Attention"),
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
         * @return {eventOut, eventQkv, eventAtt, eventScratchPad}, an event for each one of the output tensors. The order is hardcoded.
         */
        std::vector<sycl::event> Launch(
                sycl::queue &q,
                const std::vector<sycl::event> &dependencies) override {
            // dInp:  Input, then used as scratch pad.
            // dQkvr: Output, then used as input as well.
            // dAtt:  Output, then used as input as well.
            // dOut:  Output.



            constexpr int BS = 256;
            sycl::event eventOut;
            sycl::event eventPreAtt;
            sycl::event eventAtt;
            sycl::event eventScratchPad;
            sycl::event eventQkv;

            int HS = C / NH; // head size

            // STEP 1
            // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
            {
                kernels::Permute permute(
                        dQkvr + 0 * B * T * C,
                        dQkvr + 1 * B * T * C,
                        dQkvr + 2 * B * T * C,
                        dInp,
                        B, T, NH, HS);

                eventQkv = permute.Launch(q, BS, {dependencies});
            }

            const auto dQuery = dQkvr + 0 * B * T * C;
            const auto dKey = dQkvr + 1 * B * T * C;
            const auto dValue = dQkvr + 2 * B * T * C;
            auto preAtt = dInp;

            // STEP 2
            {
                const float alpha = 1.0f;
                const float beta = 0.0f; // Zero means no need for setting tensor `preAtt` to zero first.
                eventPreAtt = oneapi::mkl::blas::column_major::gemm_batch(
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
                        B * NH,
                        {eventQkv} // preAtt is used as a temporary tensor, no deps needed for it.
                );
            }


            // STEP 3 - Softmax
            {
                kernels::Softmax softmax_kernel(
                        dAtt,
                        preAtt,
                        1.0f / std::sqrt((float) HS),
                        B * NH, T
                );
                eventAtt = softmax_kernel.Launch(q, blockSizeSoftMax, {eventPreAtt});
            }

            // STEP 4 - gemm
            {
                const float alpha = 1.0f;
                const float beta = 0.0f;
                // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
                eventScratchPad = oneapi::mkl::blas::column_major::gemm_batch(
                        q,
                        oneapi::mkl::transpose::nontrans,
                        oneapi::mkl::transpose::nontrans,
                        HS, T, T,
                        alpha,
                        dValue,
                        HS, T * HS,
                        dAtt,
                        T, T * T,
                        beta,
                        dInp, // We are discarding the values inside dInp again.
                        HS, T * HS,
                        B * NH,
                        {eventQkv, eventAtt} // beta is zero again. No deps needed for tensor C.
                );
            }

            // STEP 5 - Un-permute
            {
                kernels::Unpermute unpermute_kernel(
                        dOut,
                        dInp,
                        B, T, NH, HS
                );
                eventOut = unpermute_kernel.Launch(q, BS, {eventScratchPad});
            }

            report();
            return {eventOut, eventQkv, eventAtt, eventScratchPad};
        }

    private:
        float *dOut;
        float *dQkvr;
        float *dAtt;
        float *dInp;

        const int B, T, C, NH, blockSizeSoftMax;
    };


}

