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

        std::vector<sycl::event> Launch(sycl::queue &q, int blockSize) override {
            auto event = q.submit([&](sycl::handler &h) {

                auto capturedOut = dOut;
                auto capturedInp = dInp;

                const int capturedN = this->N;
                const int capturedC = this->C;
                const float capturedInvTemp = this->invTemperature;

                h.parallel_for(
                        ///TODO: Check this coef 32 in the worksize. Is it correct?
                        sycl::nd_range<1>(
                                sycl::range<1>(Helpers::MakeDivisible(N, blockSize)),
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
                                    if (capturedInp[i*capturedC+j] > maxval) {
                                        maxval = capturedInp[i*capturedC+j];
                                    }
                                }
                                for (int j = 0; j < capturedC; j++) {
                                    sum += sycl::exp((capturedInp[i*capturedC+j] - maxval) * capturedInvTemp);
                                    capturedOut[i*capturedC+j] = (float) sum;
                                }
                                auto sumInverse = sum == 0.0 ?
                                        0.0 :
                                        1.0/sum;

                                for (int j = 0; j < capturedC; j++) {
                                    capturedOut[i*capturedC+j] *= (j > i/capturedC) ? 0 : capturedOut[i*capturedC+j] * (float)sumInverse;
                                }
                            }
                        });
            });
            report();
            return {event};
        }

    private:
        float *dOut;
        const float *dInp;

        const float invTemperature;
        const int N, C;
    };


}

