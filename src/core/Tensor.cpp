//
// Created by saleh on 14/05/24.
//
#include "npy.hpp"
#include "Tensor.h"


using namespace llmsycl::core;

template<typename T>
Tensor<T> Tensor<T>::load(std::string npyFile) {
    auto npy = npy::read_npy<T>(npyFile);
    return Tensor<T>(npy.shape, npy.data);
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
std::string Tensor<T>::getShapeStr() const{
    std::string s;
    for (auto d: getShape()) {
        s += std::to_string(d) + ", ";
    }
    return s;
}

template<typename T>
Tensor<T>::Tensor(const std::vector<size_t> &shape):
        shape(shape),
        sizeWords(getProduct(shape)) {
    hBuff = new T[sizeWords];
    std::fill(hBuff, hBuff + sizeWords, 0);
    dBuff = std::make_unique<sycl::buffer<T, 1>>(hBuff, sycl::range<1>(sizeWords));
}

template<typename T>
Tensor<T>::Tensor(std::vector<size_t> shape, std::vector<T> vecData):
        shape(shape),
        sizeWords(getProduct(shape)) {
    assert(sizeWords == vecData.size());
    hBuff = new T[sizeWords];
    std::memcpy(hBuff, vecData.data(), sizeWords * sizeof(T));
    dBuff = std::make_unique<sycl::buffer<T, 1>>(hBuff, sycl::range<1>(sizeWords));
}

template<typename T>
Tensor<T>::Tensor(std::vector<size_t> shape, const T *buff):
        shape(shape),
        sizeWords(getProduct(shape)) {
    hBuff = new T[sizeWords];
    std::memcpy(hBuff, buff, sizeWords * sizeof(T));
    dBuff = std::make_unique<sycl::buffer<T, 1>>(hBuff, sycl::range<1>(sizeWords));
}

template<typename T>
void Tensor<T>::internalSave(const std::string &npyFile) {
    npy::npy_data<T> d;

    d.data.clear();
    d.data.resize(sizeWords);
    d.data.assign(hBuff, hBuff + sizeWords);

    d.shape = shape;
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
void Tensor<T>::save(const std::string &npyFile) {
    internalSave(npyFile);
}

template<typename T>
Tensor<T>::Tensor(sycl::queue &queue, Tensor &other, bool syncHostBufferWithDevice):
        shape(other.shape),
        sizeWords(other.sizeWords) {

    // perform a device to host only if requested.
    if (syncHostBufferWithDevice) {
        other.forceD2H();
    }

    // perform a host copy
    hBuff = new T[sizeWords];
    std::memcpy(hBuff, other.hBuff, sizeWords * sizeof(T));

    // perform a device copy.
    dBuff = std::make_unique<sycl::buffer<T, 1>>(hBuff, sycl::range<1>(sizeWords));
    queue.submit([&](sycl::handler &cgh) {
        // Specify the accessors for the buffers
        auto acc1 = other.dBuff->template get_access<sycl::access::mode::read>(cgh);
        auto acc2 = dBuff->template get_access<sycl::access::mode::write>(cgh);

        // Perform the copy operation
        cgh.copy(acc1, acc2);
    });
}

template<typename T>
Tensor<T>::~Tensor() {
    // Without explicitly releasing the sycl buffer, the program will crash
    // (because sycl will try to copy stuff into hBuff which is already deleted).
    dBuff.reset();
    delete[] hBuff;
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
    auto acc = t.getAccessorHostWrite();
    for (size_t i = 0; i < t.getSize(); i++) {
        acc[i] = (float) rand() / (float)RAND_MAX;
    }
}

void llmsycl::core::fillTensorWith(Tensor<float> &t, float val) {
    auto acc = t.getAccessorHostWrite();
    for (size_t i = 0; i < t.getSize(); i++) {
        acc[i] = val;
    }
}

void llmsycl::core::fillTensorWithRandomData(Tensor<int> &t, int valUpperLimit) {
    auto acc = t.getAccessorHostWrite();
    for (size_t i = 0; i < t.getSize(); i++) {
        acc[i] = rand() % valUpperLimit;
    }
}
