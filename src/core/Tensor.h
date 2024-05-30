//
// Created by saleh on 17/05/24.
//

#pragma once

#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>
#include <cstring>
#include <memory>
#include <sycl/sycl.hpp>

namespace llmsycl::core {

    template<typename T>
    class Tensor {
    private:
        size_t getProduct(const std::vector<size_t> &shape) const;

        std::vector<size_t> maskZeros(const std::vector<size_t> &vec);

        void internalSave(const std::string &npyFile);

    protected:
        std::vector<size_t> shape;
        const size_t sizeWords = 0;

        T *__restrict hBuff = nullptr; ///< Never modify this pointer directly. Only do it through a host accessor.
        std::unique_ptr<sycl::buffer<T, 1>> dBuff;

        template<sycl::access::mode accessMode>
        inline auto getAccessorHost(size_t offset = 0) const {
            return sycl::host_accessor(*dBuff.get(), sycl::range<1>(dBuff->size() - offset), offset,
                                       sycl::mode_tag_t<accessMode>());
        }

        template<sycl::access::mode accessMode>
        inline auto getAccessorDevice(sycl::handler &h, size_t offset = 0) const {
            return sycl::accessor(*dBuff.get(), h, sycl::range<1>(dBuff->size() - offset), offset,
                                  sycl::mode_tag_t<accessMode>());
        }

    public:
        Tensor(const std::vector<size_t> &shape);

        Tensor(std::vector<size_t> shape, std::vector<T> vecData);

        Tensor(std::vector<size_t> shape, const T *buff);

        Tensor(sycl::queue &queue, Tensor &other, bool syncHostBufferWithDevice = false);

        ~Tensor();

        void save(const std::string &npyFile);

        std::vector<size_t> getShape() const;

        std::string getShapeStr() const;

        size_t getSize() const;

        /**
         * @brief A handy method to reshape the Tensor.
         * @warning The newShape should be compatible with this Tensor.
         * @note If there are elements with value "zero" in newShape, they will be inferred.
         * @warning More than one zeros in the new shape is basically useless.
         * @param newShape
         * @return
         */
        std::vector<size_t> reshape(const std::vector<size_t> &newShape);

        inline auto getAccessorHostRead(size_t offset = 0) const {
            return getAccessorHost<sycl::access::mode::read>(offset);
        }

        inline auto getAccessorHostWrite(size_t offset = 0) {
            return getAccessorHost<sycl::access::mode::write>(offset);
        }

        inline auto getAccessorHostReadWrite(size_t offset = 0) {
            return getAccessorHost<sycl::access::mode::read_write>(offset);
        }

        inline auto getAccessorDeviceRead(sycl::handler &h, size_t offset = 0) const {
            return getAccessorDevice<sycl::access::mode::read>(h, offset);
        }

        inline auto getAccessorDeviceWrite(sycl::handler &h, size_t offset = 0) {
            return getAccessorDevice<sycl::access::mode::write>(h, offset);
        }

        inline auto getAccessorDeviceReadWrite(sycl::handler &h, size_t offset = 0) {
            return getAccessorDevice<sycl::access::mode::read_write>(h, offset);
        }

        static Tensor<T> load(std::string npyFile);

        void forceD2H() {
            std::memcpy(hBuff, getAccessorHostRead().get_pointer(), sizeWords * sizeof(T));
        }

        std::vector<T> toVector() {
            std::vector<T> vec;
            vec.resize(sizeWords);
            forceD2H();
            std::memcpy(vec.data(), hBuff, sizeWords * sizeof(T));
            return vec;
        }

        void copyFromHostBlocking(sycl::queue &q, const T *hostBuff) {
            q.submit([&](sycl::handler &h) {
                auto acc = getAccessorDeviceWrite(h);
                h.copy(hostBuff, acc);
            });
            q.wait();
        }

        sycl::buffer<T, 1> &getDeviceBuff() {
            return *dBuff.get();
        }
    };



    template<typename TT>
    using TensorPtr = std::shared_ptr<Tensor<TT>>;

    void fillTensorWithRandomData(Tensor<float> &t);
    void fillTensorWith(Tensor<float> &t, float val);

    void fillTensorWithRandomData(Tensor<int> &t, int valUpperLimit);

}