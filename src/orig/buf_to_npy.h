//
// Created by saleh on 31/05/24.
//

#pragma once

#include <cuda.h>
#include "npy.hpp"

inline void writeDeviceBufToNpy(const float *buf, size_t lenWords, const std::string &filename) {
    cudaDeviceSynchronize();
    auto *hBuf = new float[lenWords];
    cudaMemcpy(hBuf, buf, lenWords * sizeof(float), cudaMemcpyDeviceToHost);

    npy::npy_data_ptr<float> d;
    d.data_ptr = hBuf;
    d.shape = {lenWords};
    d.fortran_order = false;
    npy::write_npy(filename, d);
}



inline void writeDeviceBufToNpy(const int *buf, size_t lenWords, const std::string &filename) {
    auto *hBuf = new int[lenWords];
    cudaMemcpy(hBuf, buf, lenWords * sizeof(int), cudaMemcpyDeviceToHost);

    npy::npy_data_ptr<int> d;
    d.data_ptr = hBuf;
    d.shape = {lenWords};
    d.fortran_order = false;
    npy::write_npy(filename, d);
}


