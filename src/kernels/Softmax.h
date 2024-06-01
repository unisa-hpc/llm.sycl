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
                int N, int C
        ) :
                BaseKernel("Softmax"),
                tnOut(tnOut),
                outOffset(outOffset),
                tnInp(tnInp),
                inpOffset(inpOffset),
                invTemperature(invTemperature),
                N(N), C(C) {

            addTensorDetailsToReport("tnOut", tnOut);
            addTensorDetailsToReport("tnInp", tnInp);
            addScalarParamToReport("invTemperature", invTemperature);
            addScalarParamToReport("N", N);
            addScalarParamToReport("C", C);

        }

        sycl::event Launch(sycl::queue &q, int blockSize) override {
            auto event = q.submit([&](sycl::handler &h) {

                auto accTnOut = tnOut.getAccessorDeviceWrite(h, outOffset);
                auto accTnInp = tnInp.getAccessorDeviceRead(h, inpOffset);

                const int capturedN = this->N;
                const int capturedC = this->C;
                const float capturedInvTemp = this->invTemperature;

                h.parallel_for(
                        ///TODO: Check this coef 32 in the worksize. Is it correct?
                        sycl::nd_range<1>(
                                sycl::range<1>(Helpers::MakeDivisible(N*C, blockSize)),
                                sycl::range<1>(blockSize)
                        ),
                        [=](sycl::nd_item<1> item) {

                            // const size_t subgroup_size = item.get_sub_group().get_local_range().get(0); // Get the size of the subgroup
                            // const size_t num_sg_in_wg = item.get_local_range(0) / subgroup_size; // num warps in the workgroup
                            // const size_t local_id = item.get_local_id(0); // Get the local ID of the work-item
                            // const size_t subgroup_index = local_id / subgroup_size; // Calculate the index of the subgroup
                            // const size_t idInSg = local_id % subgroup_size; // Calculate the index of the work-item within the subgroup

                            // inp is (N, T)
                            // out is (N, T), each row of inp will get softmaxed
                            const auto i = item.get_global_id(0);
                            if (i < capturedN) {
                                float maxval = -INFINITY;
                                double sum = 0.0;
                                for (int j = 0; j < capturedC; j++) {
                                    float maxval_prev = maxval;
                                    if (accTnInp[i*capturedC+j] > maxval) {
                                        maxval = accTnInp[i*capturedC+j];
                                        sum = sum *
                                                sycl::exp((maxval_prev - maxval) * capturedInvTemp) +
                                                sycl::exp((accTnInp[i*capturedC+j] - maxval) * capturedInvTemp);
                                    }
                                    else {
                                        sum += sycl::exp((accTnInp[i*capturedC+j] - maxval) * capturedInvTemp);
                                    }
                                }

                                for (int j = 0; j < capturedC; j++) {
                                    accTnOut[i*capturedC+j] =
                                            sycl::exp((accTnInp[i*capturedC+j] - maxval) * capturedInvTemp) / sum;
                                }
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
        const int N, C;
    };


}

