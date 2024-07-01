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
                float *dOut,
                const float *dInp,
                float invTemperature,
                int N, int C
        ) :
                BaseKernel("Softmax"),
                dOut(dOut),
                dInp(dInp),
                invTemperature(invTemperature),
                N(N), C(C) {

            addScalarParamToReport("invTemperature", invTemperature);
            addScalarParamToReport("N", N);
            addScalarParamToReport("C", C);
        }

        sycl::event Launch(
                sycl::queue &q,
                int blockSize,
                const std::vector<sycl::event> &dependencies) override {

            auto event = q.submit([&](sycl::handler &h) {

                h.depends_on(dependencies);

                auto capturedOut = dOut;
                auto capturedInp = dInp;
                const int capturedN = this->N;
                const int capturedC = this->C;
                const float capturedInvTemp = this->invTemperature;

                assert(capturedC % 4  == 0);
                //sycl::stream os(10240, 1280, h);

                h.parallel_for(
                        ///TODO: Check this coef 32 in the worksize. Is it correct?
                        sycl::nd_range<1>(
                                sycl::range<1>(Helpers::MakeDivisible(N*C*32, blockSize)),
                                sycl::range<1>(blockSize)
                        ),
                        [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(32)]] {

                            const int WARP_SIZE = item.get_sub_group().get_local_range()[0];
                            int lane_id = item.get_local_id() % WARP_SIZE;
                            int warp_id = item.get_local_id() / WARP_SIZE;
                            int num_warps = blockSize / WARP_SIZE;

                            // micro-optimization: we iterate backwards so that
                            // after the softmax backward operation completes, the cache retains the
                            // part of the matrix close to the upper left corner, which benefits the
                            // matmul operation that immediately follows.
                            //int idx = 	item.get_group(0) * num_warps + warp_id; // forward order

                            int idx = (item.get_group_range(0) - item.get_group(0) - 1) * num_warps + warp_id; // backward order

                            //if (idx == 0) os << "idx=" << idx << " lane_id=" << lane_id << " warp_id=" << warp_id << " num_warps=" << num_warps << sycl::endl;

                            if(idx >= capturedN * capturedC) {
                                return;
                            }
                            int own_pos = idx % capturedC;
                            int pos_by_4 = own_pos / 4;

                            // one row of inp, i.e. inp[idx, :] of shape (capturedC,)
                            auto x = capturedInp + idx * capturedC;

                            // not INF, so we don't get NaNs accidentally when subtracting two values.
                            const float flt_max = FLT_MAX; // to avoid including float.h
                            float maxval = -flt_max;
                            float sumval = 0.0f;

                            //const float* x_aligned = reinterpret_cast<const float*>(__builtin_assume_aligned(x, 16));
                            for (int i = lane_id; i < pos_by_4; i += WARP_SIZE) {
                                float regarray[4];
                                for (int k = 0; k < 4; ++k) {
                                    regarray[k] = (float)x[4*i + k];
                                }
                                float old_maxval = maxval;
                                for(int k = 0; k < 4; ++k) {
                                    maxval = sycl::fmax(maxval, regarray[k]);
                                }
                                sumval *= sycl::exp(capturedInvTemp * (old_maxval - maxval));
                                for(int k = 0; k < 4; ++k) {
                                    sumval += sycl::exp(capturedInvTemp * (regarray[k] - maxval));
                                }
                            }

                            if(4*pos_by_4 + lane_id <= own_pos) {
                                float old_maxval = maxval;
                                maxval = sycl::fmax(maxval, (float)x[4*pos_by_4 + lane_id]);
                                sumval *= sycl::exp(capturedInvTemp * (old_maxval - maxval));
                                sumval += sycl::exp(capturedInvTemp * ((float)x[4*pos_by_4 + lane_id] - maxval));
                            }

                            float global_maxval = sycl::reduce_over_group(item.get_sub_group(), maxval, sycl::maximum<>());
                            sumval *= sycl::exp(capturedInvTemp * (maxval - global_maxval));

                            float sum = sycl::reduce_over_group(item.get_sub_group(), sumval, sycl::plus<>());
                            //if (idx == 0) os << "sum[" << idx << "]=" << sum << sycl::endl;

                            float norm = 1.f / sum;

                            // divide the whole row by the sum
                            for (int i = lane_id; i <= own_pos; i += WARP_SIZE) {
                                // recalculation is faster than doing the round-trip through memory.
                                float ev = sycl::exp(capturedInvTemp * (x[i] - global_maxval));
                                /*{
                                    // DEBUG
                                    auto iidx0 = idx * capturedC + i;
                                    if (idx==0)
                                        os << "idx=" << idx << " i=" << i << " own_pos=" << own_pos << " pos_by_4=" << pos_by_4 << " lane_id=" << lane_id << " warp_id=" << warp_id << " num_warps=" << num_warps << " maxval=" << maxval << " sumval=" << sumval << " global_maxval=" << global_maxval << " sum=" << sum << " norm=" << norm << sycl::endl;
                                    if (iidx0 == 0)
                                    {
                                        os << "capturedOut @ " << iidx0 << ": " << "ev=" <<ev << " norm=" << norm << "ev*norm=" << ev*norm << sycl::endl;
                                    }
                                }*/
                                capturedOut[idx * capturedC + i] = ev * norm;
                            }

                        });
            });

            report();
            return event;
        }

    private:
        float *dOut;
        const float *dInp;

        const float invTemperature;
        const int N, C;
    };


}

