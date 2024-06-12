//
// Created by saleh on 23/05/24.
//

#pragma once

#include <sycl/sycl.hpp>

#include "core/Tensor.h"
#include "BaseKernel.h"

namespace llmsycl::kernels {

    class LayerNorm : public BaseKernel {
        friend class sycl::handler;

    public:
        LayerNorm(
                float *dOut,
                float *dMean,
                float *dRstd,
                const float *dInp,
                const float *dWeight,
                const float *dBias,
                int B, int T, int C
        ) :
                BaseKernel("LayerNorm"),
                dOut(dOut),
                dMean(dMean),
                dRstd(dRstd),
                dInp(dInp),
                dWeight(dWeight),
                dBias(dBias),
                B(B), T(T), C(C) {
            addScalarParamToReport("B", B);
            addScalarParamToReport("T", T);
            addScalarParamToReport("C", C);
        }

        std::vector<sycl::event> Launch(
                sycl::queue &q,
                int blockSize,
                const std::vector<sycl::event> &dependencies) override {

            auto event = q.submit([&](sycl::handler &h) {

                h.depends_on(dependencies);

                auto capturedOut = dOut;
                auto capturedMean = dMean;
                auto capturedRstd = dRstd;
                auto capturedInp = dInp;
                auto capturedWeight = dWeight;
                auto capturedBias = dBias;

                const int capturedN = this->B * this->T;
                const int capturedC = this->C;
                if (capturedC % blockSize != 0) {
                    throw std::runtime_error("LayerNorm: C must be divisible to the block size.");
                }

                sycl::local_accessor<float, 1> localSliceC(capturedC, h);
                sycl::local_accessor<float, 1> localW(capturedC, h);
                sycl::local_accessor<float, 1> localB(capturedC, h);

                sycl::stream os(10240, 1280, h);

                h.parallel_for(
                        sycl::nd_range<1>(
                                sycl::range<1>(B * T * blockSize),
                                sycl::range<1>(blockSize)
                        ),
                        /*
                        [=](sycl::nd_item<1> item) {
                            const int idx = (int) item.get_global_id(0);
                            if (idx < capturedN) {
                                float eps = 1e-5f;

                                // calculate the mean
                                float m = 0.0f;
                                for (int i = 0; i < capturedC; i++) {
                                    m += capturedInp[idx * capturedC + i];
                                }
                                m = m / (float)capturedC;
                                os << m << sycl::endl;


                                // calculate the variance (without any bias correction)
                                float v = 0.0f;
                                for (int i = 0; i < capturedC; i++) {
                                    float xshift = capturedInp[idx * capturedC + i] - m;
                                    v += xshift * xshift;
                                }
                                v = v / (float)capturedC;
                                // calculate the rstd
                                float s = 1.0f / sycl::sqrt(v + eps);


                                for (int i = 0; i < capturedC; i++) {
                                    float n = (s * (capturedInp[idx * capturedC + i] - m)); // normalized output
                                    float o = n * capturedWeight[i] + capturedBias[i]; // scale and shift it
                                    capturedOut[idx * capturedC + i] = o; // write
                                }
                                // cache the mean and rstd for the backward pass later
                                capturedMean[idx] = m;
                                capturedRstd[idx] = s;
                            }
                        });
                        */


                        [=](sycl::nd_item<1> item) {
                            // Let's assume every thread block handles one slice of size C.
                            // Later we can extend it using block-stride loops.
                            const int tid = (int) item.get_local_id(0);
                            const int grid_index = (int)item.get_group(0);
                            if (grid_index >= capturedN) {
                                os << "This should not have happened!" << sycl::endl;
                            }



                            // Stage 1. Cache the input slice into the local memory.
                            for (int i = tid; i < capturedC; i += blockSize) {
                                localSliceC[i] = capturedInp[grid_index*capturedC +i];
                                localW[i] = capturedWeight[i];
                                localB[i] = capturedBias[i];
                            }
                            item.barrier();


                            // Stage 2. Calculate the mean and variance.
                            float m = 0.0f;
                            float v = 0.0f;

                            float sum = 0;
                            for (int i = tid; i < capturedC; i += blockSize) {
                                float ss = sycl::reduce_over_group(item.get_group(), localSliceC[i], sycl::plus<float>());
                                sum += ss;
                                //item.barrier();
                            }
                            m = sum / (float)capturedC;
                            //os << m << sycl::endl;


                            // Stage 3. Calculate the variance
                            for (int i = tid; i < capturedC; i += blockSize) {
                                float xshift = localSliceC[i] - m;
                                float x2 = xshift * xshift;
                                float ss = sycl::reduce_over_group(item.get_group(), x2, sycl::plus<float>());
                                v += ss;
                            }
                            v = v / (float)capturedC;
                            float s = 1.0f / sycl::sqrt(v + 1e-5f);
                            //os << v << sycl::endl;


                            // Stage 5. Calculate the output.
                            for (int i = tid; i < capturedC; i += blockSize) {
                                float n = s * (localSliceC[i] - m); // normalized
                                float o = n * localW[i] + localB[i]; // scale and shift it
                                capturedOut[grid_index * capturedC + i] = o; // write
                                //os <<  i << "=" << o << sycl::endl;
                                //item.barrier();
                            }
                            capturedMean[grid_index] = m;
                            capturedRstd[grid_index] = s;
                        });


            });
            report();
            return {event};
        }

    private:
        float *dOut;
        float *dMean;
        float *dRstd;
        const float *dInp;
        const float *dWeight;
        const float *dBias;
        const int B, T, C;
    };


}

