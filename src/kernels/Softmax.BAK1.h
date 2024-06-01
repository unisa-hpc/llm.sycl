//
// Created by saleh on 23/05/24.
//

#pragma once

#include <sycl/sycl.hpp>
#include <cassert>

#include "core/Tensor.h"
#include "BaseKernel.h"
#include <climits>
#include <cfloat>

namespace llmsycl::kernels {

    class Softmax : public BaseKernel {
        friend class sycl::handler;

    public:
        Softmax(
                core::Tensor<float> &tnOut,
                size_t outOffset,
                core::Tensor<float> &tnInp,
                size_t inpOffset,
                float invTemperature,
                int N, int T
        ) :
                BaseKernel("Softmax"),
                tnOut(tnOut),
                outOffset(outOffset),
                tnInp(tnInp),
                inpOffset(inpOffset),
                invTemperature(invTemperature),
                N(N), T(T) {

            addTensorDetailsToReport("tnOut", tnOut);
            addTensorDetailsToReport("tnInp", tnInp);
            addScalarParamToReport("invTemperature", invTemperature);
            addScalarParamToReport("N", N);
            addScalarParamToReport("T", T);

        }

        sycl::event Launch(sycl::queue &q, int blockSize) override {
            auto event = q.submit([&](sycl::handler &h) {

                auto accTnOut = tnOut.getAccessorDeviceWrite(h, outOffset);
                auto accTnInp = tnInp.getAccessorDeviceRead(h, inpOffset);

                const int capturedN = this->N;
                const int capturedT = this->T;
                const float capturedInvTemp = this->invTemperature;

                h.parallel_for(
                        ///TODO: Check this coef 32 in the worksize. Is it correct?
                        sycl::nd_range<1>(
                                sycl::range<1>(Helpers::MakeDivisible(N * T * 32, blockSize)),
                                sycl::range<1>(blockSize)
                        ),
                        [=](sycl::nd_item<1> item) {

                            const size_t subgroup_size = item.get_sub_group().get_local_range().get(0); // Get the size of the subgroup
                            const size_t num_sg_in_wg = item.get_local_range(0) / subgroup_size; // num warps in the workgroup
                            const size_t local_id = item.get_local_id(0); // Get the local ID of the work-item
                            const size_t subgroup_index = local_id / subgroup_size; // Calculate the index of the subgroup
                            const size_t idInSg = local_id % subgroup_size; // Calculate the index of the work-item within the subgroup


                            // inp, out shape: (N, T, T), where N = B * NH
                            // fuses the multiplication by scale inside attention
                            // directly autoregressive, so we only compute the lower triangular part
                            // uses the online softmax algorithm
                            if (capturedT % 4 != 0) {
                                return;
                            }

                            // micro-optimization: we iterate backwards so that
                            // after the softmax backward operation completes, the cache retains the
                            // part of the matrix close to the upper left corner, which benefits the
                            // matmul operation that immediately follows.
                            // int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank(); // forward order
                            size_t idx = (	item.get_group_range(0) - item.get_group(0) - 1)
                                    * num_sg_in_wg
                                    + subgroup_index; // backward order
                            if(idx >= capturedN * capturedT) {
                                return;
                            }
                            size_t own_pos = idx % capturedT;
                            size_t pos_by_4 = own_pos / 4;

                            // one row of inp, i.e. inp[idx, :] of shape (T,)
                            //const float* x = inp + idx * T;

                            // not INF, so we don't get NaNs accidentally when subtracting two values.
                            float maxval = -FLT_MAX;
                            float sumval = 0.0f;


                            for (auto i = idInSg; i < pos_by_4; i += subgroup_size) {
                                float val0 = accTnInp[4* idx * capturedT + i*4 + 0];
                                float val1 = accTnInp[4* idx * capturedT + i*4 + 1];
                                float val2 = accTnInp[4* idx * capturedT + i*4 + 2];
                                float val3 = accTnInp[4* idx * capturedT + i*4 + 3];
                                float old_maxval = maxval;
                                maxval = maxval >= val0 ? maxval : val0;
                                maxval = maxval >= val1 ? maxval : val1;
                                maxval = maxval >= val2 ? maxval : val2;
                                maxval = maxval >= val3 ? maxval : val3;

                                sumval *= sycl::exp(capturedInvTemp * (old_maxval - maxval));
                                sumval += sycl::exp(capturedInvTemp * (val0 - maxval));
                                sumval += sycl::exp(capturedInvTemp * (val1 - maxval));
                                sumval += sycl::exp(capturedInvTemp * (val2 - maxval));
                                sumval += sycl::exp(capturedInvTemp * (val3 - maxval));
                            }

                            if(4*pos_by_4 + idInSg <= own_pos) {
                                float old_maxval = maxval;
                                auto vv = accTnInp[4* idx * capturedT + 4*pos_by_4 + idInSg];
                                maxval = maxval >= vv ? maxval : vv;
                                sumval *= sycl::exp(capturedInvTemp * (old_maxval - maxval));
                                sumval += sycl::exp(capturedInvTemp * (accTnInp[idx * capturedT + 4*pos_by_4 + idInSg] - maxval));
                            }

                            float global_maxval = sycl::reduce_over_group(item.get_sub_group(), maxval, sycl::maximum<>());
                            sumval *= sycl::exp(capturedInvTemp * (maxval - global_maxval));

                            float sum = sycl::reduce_over_group(item.get_sub_group(), sumval, sycl::plus<>());
                            float norm = 1.f / sum;

                            // divide the whole row by the sum
                            for (auto i = idInSg; i <= own_pos; i += subgroup_size) {
                                // recalculation is faster than doing the round-trip through memory.
                                float ev = sycl::exp(capturedInvTemp * ( accTnInp[idx * capturedT + i] - global_maxval));
                                accTnOut[idx * capturedT + i] = ev * norm;
                            }


                        });
            });
            report();
            return event;
        }

    private:
        core::Tensor<float> &tnOut;
        const size_t outOffset;
        core::Tensor<float> &tnInp;
        const size_t inpOffset;

        const float invTemperature;
        const int N, T;
    };


}

