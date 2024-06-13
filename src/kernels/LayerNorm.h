//
// Created by saleh on 23/05/24.
//

#pragma once

#include <sycl/sycl.hpp>

#include "core/Tensor.h"
#include "BaseKernel.h"

#undef USE_KERNEL_1_SLICE_C_PER_BLOCK
#undef USE_KERNEL_1_SLICE_C_PER_WARP
#define USE_KERNEL_1_SLICE_C_PER_WARP_SMEM


namespace llmsycl::kernels {

    class LayerNorm : public BaseKernel {
        friend class sycl::handler;

    public:
        LayerNorm(
                float * __restrict__ dOut,
                float * __restrict__ dMean,
                float * __restrict__ dRstd,
                const float * __restrict__ dInp,
                const float *__restrict__ dWeight,
                const float * __restrict__ dBias,
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

                sycl::stream os(10240, 1280, h);

#ifdef USE_KERNEL_1_SLICE_C_PER_BLOCK
                sycl::local_accessor<float, 1> localSliceC(capturedC, h);
                sycl::local_accessor<float, 1> localW(capturedC, h);
                sycl::local_accessor<float, 1> localB(capturedC, h);
                h.parallel_for(
                        sycl::nd_range<1>(
                                sycl::range<1>(B * T * blockSize),
                                sycl::range<1>(blockSize)
                        ),
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
#endif

#ifdef USE_KERNEL_1_SLICE_C_PER_WARP
                h.parallel_for(
                        sycl::nd_range<1>(
                                sycl::range<1>(B * T * 32),
                                sycl::range<1>(blockSize)
                        ),
                        [=](sycl::nd_item<1> item) {
                            const int warp_size = item.get_sub_group().get_local_range().get(0);
                            const int sid = (int) item.get_sub_group().get_local_id();
                            const int warp_id_group = item.get_local_id(0) / warp_size; // warp id in current block.
                            const int group_size_in_warps = item.get_local_range(0) / warp_size; // how many warps per block
                            const int warp_id_global = item.get_group(0) * group_size_in_warps + warp_id_group; // warp id in the grid.

                            /*
                            if (item.get_global_id() == 0) {
                                os << "warp_size=" << warp_size << sycl::endl;
                                os << "sid=" << sid << sycl::endl;
                                os << "warp_id_group=" << warp_id_group << sycl::endl;
                                os << "group_size_in_warps=" << group_size_in_warps << sycl::endl;
                                os << "warp_id_global=" << warp_id_global << sycl::endl;
                            }
                            */


                            if (warp_id_global >= capturedN) {
                                os << "This should not have happened!" << sycl::endl;
                            }


                            // Stage 2. Calculate the mean and variance.
                            float m = 0.0f;
                            float v = 0.0f;

                            auto pInSlice = capturedInp + warp_id_global * capturedC;

                            float sum = 0;
                            for (int i = sid; i < capturedC; i += warp_size) {
                                sum += pInSlice[i];
                            }
                            sum = sycl::reduce_over_group(item.get_sub_group(), sum, sycl::plus<float>());
                            m = sum / (float)capturedC;
                            //os << m << sycl::endl;


                            // Stage 3. Calculate the variance
                            for (int i = sid; i < capturedC; i += warp_size) {
                                float xshift = pInSlice[i] - m;
                                v += xshift * xshift;
                            }
                            v = sycl::reduce_over_group(item.get_sub_group(), v, sycl::plus<float>());
                            v = v / (float)capturedC;
                            float s = 1.0f / sycl::sqrt(v + 1e-5f);
                            //os << v << sycl::endl;


                            // Stage 5. Calculate the output.
                            for (int c = sid; c < capturedC; c += warp_size) {
                                float n = s * (pInSlice[c] - m); // normalized
                                float o = n * capturedWeight[c] + capturedBias[c]; // scale and shift it
                                capturedOut[warp_id_global * capturedC + c] = o; // write
                                //os <<  i << "=" << o << sycl::endl;
                                //item.barrier();
                            }
                            capturedMean[warp_id_global] = m;
                            capturedRstd[warp_id_global] = s;

                        });
            });
#endif

#ifdef USE_KERNEL_1_SLICE_C_PER_WARP_SMEM
                sycl::local_accessor<float, 1> localW(capturedC, h);
                sycl::local_accessor<float, 1> localB(capturedC, h);
                h.parallel_for(
                        sycl::nd_range<1>(
                                sycl::range<1>(B * T * 32),
                                sycl::range<1>(blockSize)
                        ),
                        [=](sycl::nd_item<1> item) {
                            const int warp_size = item.get_sub_group().get_local_range().get(0);
                            const int sid = (int) item.get_sub_group().get_local_id();
                            const int warp_id_group = item.get_local_id(0) / warp_size; // warp id in current block.
                            const int group_size_in_warps = item.get_local_range(0) / warp_size; // how many warps per block
                            const int warp_id_global = item.get_group(0) * group_size_in_warps + warp_id_group; // warp id in the grid.

                            const int tid = (int) item.get_local_id(0);
                            const int grid_index = (int)item.get_group(0);

                            /*
                            if (item.get_global_id() == 0) {
                                os << "warp_size=" << warp_size << sycl::endl;
                                os << "sid=" << sid << sycl::endl;
                                os << "warp_id_group=" << warp_id_group << sycl::endl;
                                os << "group_size_in_warps=" << group_size_in_warps << sycl::endl;
                                os << "warp_id_global=" << warp_id_global << sycl::endl;
                            }
                            */

                            if (warp_id_global >= capturedN) {
                                os << "This should not have happened!" << sycl::endl;
                            }

                            // Stage 1. Loading into the shared memory.

                            // We cannot use smem for the input tensor.
                            // Each block has many warps. each warp needs a slice of size C.
                            // So, we need a lot of smem just for the input tensor. Better to leave it.
                            for (int i = tid; i < capturedC; i += blockSize) {
                                localW[i] = capturedWeight[i];
                                localB[i] = capturedBias[i];
                            }

                            // No need to sync at block level.
                            // Our threads are working as a group in warps.
                            // Although they share the same block.
                            item.get_sub_group().barrier(); // __syncwarp();





                            // Stage 2. Calculate the mean and variance.
                            float m = 0.0f;
                            float v = 0.0f;

                            auto pInSlice = capturedInp + warp_id_global * capturedC;

                            float sum = 0;
                            for (int i = sid; i < capturedC; i += warp_size) {
                                sum += pInSlice[i];
                            }
                            sum = sycl::reduce_over_group(item.get_sub_group(), sum, sycl::plus<float>());
                            m = sum / (float)capturedC;
                            //os << m << sycl::endl;


                            // Stage 3. Calculate the variance
                            for (int i = sid; i < capturedC; i += warp_size) {
                                float xshift = pInSlice[i] - m;
                                v += xshift * xshift;
                            }
                            v = sycl::reduce_over_group(item.get_sub_group(), v, sycl::plus<float>());
                            v = v / (float)capturedC;
                            float s = 1.0f / sycl::sqrt(v + 1e-5f);
                            //os << v << sycl::endl;


                            // Stage 5. Calculate the output.
                            for (int c = sid; c < capturedC; c += warp_size) {
                                float n = s * (pInSlice[c] - m); // normalized
                                float o = n * localW[c] + localB[c]; // scale and shift it
                                capturedOut[warp_id_global * capturedC + c] = o; // write
                                //os <<  i << "=" << o << sycl::endl;
                                //item.barrier();
                            }
                            capturedMean[warp_id_global] = m;
                            capturedRstd[warp_id_global] = s;

                        });
            });
#endif


            report();
            return {event};
        }

    private:
        float * __restrict__ dOut;
        float * __restrict__ dMean;
        float * __restrict__ dRstd;
        const float * __restrict__ dInp;
        const float * __restrict__ dWeight;
        const float * __restrict__ dBias;
        const int B, T, C;
    };


}

