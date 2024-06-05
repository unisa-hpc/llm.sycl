//
// Created by saleh on 14/05/24.
//
#include "npy.hpp"
#include "Tensor.h"


using namespace llmsycl::core;

template<typename T>
Tensor<T> Tensor<T>::loadToHost(sycl::queue &queue, std::string npyFile) {
    auto npy = npy::read_npy<T>(npyFile);
    return Tensor<T>(queue, npy.shape, npy.data);
}

template<typename T>
size_t Tensor<T>::getSize() const {
    return getProduct(getShape());
}

template<typename T>
size_t Tensor<T>::getProduct(const std::vector<size_t> &shape) const {
    return std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>{});
}

template<typename T>
std::vector<size_t> Tensor<T>::maskZeros(const std::vector<size_t> &vec) {
    std::vector<size_t> r;
    for (auto e: vec) {
        if (e != 0) {
            r.push_back(e);
        }
    }
    return r;
}

template<typename T>
std::string Tensor<T>::getShapeStr() const {
    std::string s;
    for (auto d: getShape()) {
        s += std::to_string(d) + ", ";
    }
    return s;
}

template<typename T>
Tensor<T>::Tensor(sycl::queue &queue, const std::vector<size_t> &shape):
        shape(shape),
        sizeWords(getProduct(shape)),
        queue(queue) {
    // Malloc_host copies data elements through PCIe on each access, so it is slower when there are many accesses.
    // It could be faster when you only access 10 elements for example, and then you dont need to copy the whole data.
    // malloc_shared on the other hand, is managed by the compiler and it decides where to move the data around (as a whole).
    // There is also a prefetch flag for malloc_shared to prefetch the data to the device.
    // This could improve the performance but it could also hurt even more if the prefetched data is wrong or not needed.
    // The safest bet is ALWAYS manual data movement between host and device.
    // So, use C++ heap or if you want pinned memory; along with sycl::malloc_device.
    // and move data with queue.memcpy().
    // Just like in CUDA.
    hBuff = new T[sizeWords]; // Dont use sycl::malloc_host.
    std::fill(hBuff, hBuff + sizeWords, 0);
    dBuff = static_cast<T *>(sycl::malloc_device(sizeWords * sizeof(T), queue));
}

template<typename T>
Tensor<T>::Tensor(sycl::queue &queue, std::vector<size_t> shape, std::vector<T> vecData):
        shape(shape),
        sizeWords(getProduct(shape)),
        queue(queue)  {
    assert(sizeWords == vecData.size());
    hBuff = new T[sizeWords];
    std::memcpy(hBuff, vecData.data(), sizeWords * sizeof(T));
    dBuff = static_cast<T *>(sycl::malloc_device(sizeWords * sizeof(T), queue));
    syncBlockingH2D();
}

template<typename T>
Tensor<T>::Tensor(sycl::queue &queue, std::vector<size_t> shape, const T *buff):
        shape(shape),
        sizeWords(getProduct(shape)),
        queue(queue)  {
    hBuff = new T[sizeWords];
    std::memcpy(hBuff, buff, sizeWords * sizeof(T));
    dBuff = static_cast<T *>(sycl::malloc_device(sizeWords * sizeof(T), queue));
    syncBlockingH2D();
}

template<typename T>
Tensor<T>::Tensor(Tensor &other, bool fromItsDeviceBuffer):
        shape(other.shape),
        sizeWords(other.sizeWords),
        queue(other.queue){

    // The idea is not to touch the other Tensor when we are creating one from it!
    dBuff = static_cast<T *>(sycl::malloc_device(sizeWords * sizeof(T), queue));
    if (fromItsDeviceBuffer) {
        queue.memcpy(dBuff, other.dBuff, sizeWords * sizeof(T));
    }
    hBuff = new T[sizeWords];
    if (fromItsDeviceBuffer) {
        syncBlockingD2H();
    } else {
        std::memcpy(hBuff, other.hBuff, sizeWords * sizeof(T));
        syncBlockingH2D();
    }
}

template<typename T>
void Tensor<T>::internalSaveHostToNpy(size_t offset, size_t lenWords, const std::string &npyFile) {
    npy::npy_data<T> d;

    if (offset + lenWords > sizeWords) {
        throw std::invalid_argument("The given offset and lenWords are out of the range of this Tensor.");
    }

    d.data.clear();
    d.data.resize(lenWords);
    d.data.assign(hBuff + offset, hBuff + offset + lenWords);

    ///TODO: Add the shape to the npy file.
    d.shape = {lenWords};
    d.fortran_order = false; // We don't want col-major.
    npy::write_npy(npyFile, d);
}

template<typename T>
std::vector<size_t> Tensor<T>::getShape() const {
    return shape;
}

template<typename T>
std::vector<size_t> Tensor<T>::reshape(const std::vector<size_t> &newShape) {
    auto cloned = newShape;
    for (auto &d: cloned) {
        if (d == 0) {
            auto t1 = getProduct(getShape());
            auto t2 = getProduct(maskZeros(cloned));
            if (t1 % t2 != 0) {
                throw std::invalid_argument("The given shape is incompatible with this Tensor. Reshaping failed.");
            }
            d = t1 / t2;
        }
    }
    if (getProduct(getShape()) != getProduct(cloned)) {
        throw std::invalid_argument("The given shape is incompatible with this Tensor. Reshaping failed.");
    }
    shape = cloned;
    return shape;
}

template<typename T>
void Tensor<T>::saveHostToNpy(const std::string &npyFile) {
    internalSaveHostToNpy(0, sizeWords, npyFile);
}

template<typename T>
void Tensor<T>::saveHostToNpy(size_t offset, size_t lenWords, const std::string &npyFile) {
    internalSaveHostToNpy(offset, lenWords, npyFile);
}

template<typename T>
Tensor<T>::~Tensor() {
    delete [] hBuff;
    sycl::free(dBuff, queue);
}

template
class llmsycl::core::Tensor<int>;

template
class llmsycl::core::Tensor<unsigned>;

template
class llmsycl::core::Tensor<size_t>;

template
class llmsycl::core::Tensor<float>;

void llmsycl::core::fillTensorWithRandomData(Tensor<float> &t) {
    auto p = t.getHostBuffer();
    for (size_t i = 0; i < t.getSize(); i++) {
        p[i] = (float) rand() / (float) RAND_MAX;
    }
    t.syncBlockingH2D();
}

void llmsycl::core::fillTensorWith(Tensor<float> &t, float val) {
    auto p = t.getHostBuffer();
    for (size_t i = 0; i < t.getSize(); i++) {
        p[i] = val;
    }
    t.syncBlockingH2D();
}

void llmsycl::core::fillTensorWithRandomData(Tensor<int> &t, int valUpperLimit) {
    auto p = t.getHostBuffer();
    for (size_t i = 0; i < t.getSize(); i++) {
        p[i] = rand() % valUpperLimit;
    }
    t.syncBlockingH2D();
}

void llmsycl::core::saveFromDeviceToNpy(sycl::queue &q, const float *dBuf, size_t lenWords, const std::string &filename) {
    q.wait_and_throw();
    auto *hBuf = new float[lenWords];
    q.memcpy(hBuf, dBuf, lenWords * sizeof(float)).wait();

    npy::npy_data_ptr<float> d;
    d.data_ptr = hBuf;
    d.shape = {lenWords};
    d.fortran_order = false;
    npy::write_npy(filename, d);
    delete[] hBuf;
}
