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
    class tensor {
    private:
        size_t getProduct(const std::vector<size_t> &shape);

        std::vector<size_t> maskZeros(const std::vector<size_t> &vec);

        void internalSave(const std::string &npyFile);

    protected:
        std::vector<size_t> shape;
        const size_t sizeWords = 0;

        T *__restrict hBuff = nullptr; ///< Never modify this pointer directly. Only do it through a host accessor.
        std::unique_ptr<sycl::buffer<T, 1>> dBuff;

        template<sycl::access::mode accessMode>
        inline auto getAccessorHost() const {
            return sycl::host_accessor(*dBuff.get(), sycl::mode_tag_t<accessMode>());
        }

        template<sycl::access::mode accessMode>
        inline auto getAccessorDevice(sycl::handler &h) const {
            return sycl::accessor(*dBuff.get(), h, sycl::mode_tag_t<accessMode>());
        }

    public:
        tensor(const std::vector<size_t> &shape);

        tensor(std::vector<size_t> shape, std::vector<T> vecData);

        tensor(sycl::queue &queue, tensor &other, bool syncHostBufferWithDevice=false);

        ~tensor();

        void save(const std::string &npyFile);

        std::vector<size_t> getShape();

        std::string getShapeStr();

        size_t getSize();

        /**
         * @brief A handy method to reshape the tensor.
         * @warning The newShape should be compatible with this tensor.
         * @note If there are elements with value "zero" in newShape, they will be inferred.
         * @warning More than one zeros in the new shape is basically useless.
         * @param newShape
         * @return
         */
        std::vector<size_t> reshape(const std::vector<size_t> &newShape);

        inline auto getAccessorHostRead() const {
            return getAccessorHost<sycl::access::mode::read>();
        }

        inline auto getAccessorHostWrite() {
            return getAccessorHost<sycl::access::mode::write>();
        }

        inline auto getAccessorHostReadWrite() {
            return getAccessorHost<sycl::access::mode::read_write>();
        }

        inline auto getAccessorDeviceRead(sycl::handler &h) const {
            return getAccessorDevice<sycl::access::mode::read>(h);
        }

        inline auto getAccessorDeviceWrite(sycl::handler &h) {
            return getAccessorDevice<sycl::access::mode::write>(h);
        }

        inline auto getAccessorDeviceReadWrite(sycl::handler &h) {
            return getAccessorDevice<sycl::access::mode::read_write>(h);
        }

        static tensor<T> load(std::string npyFile);

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
    };




}