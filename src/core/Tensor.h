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

        void internalSaveHostToNpy(size_t offset, size_t lenWords, const std::string &npyFile);

    protected:
        std::vector<size_t> shape;
        const size_t sizeWords = 0;
        sycl::queue &queue;

        T * hBuff = nullptr;
        T * dBuff = nullptr;

    public:
        Tensor(sycl::queue &queue, const std::vector<size_t> &shape);

        Tensor(sycl::queue &queue, std::vector<size_t> shape, std::vector<T> vecData);

        Tensor(sycl::queue &queue, std::vector<size_t> shape, const T *buff);

        Tensor(Tensor &other, bool fromItsDeviceBuffer = false);

        ~Tensor();

        void saveHostToNpy(const std::string &npyFile);

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

        static Tensor<T> loadToHost(sycl::queue &queue, std::string npyFile);

        void syncBlockingD2H() {
            queue.memcpy(hBuff, dBuff, sizeWords * sizeof(T));
        }

        void syncBlockingH2D() {
            queue.memcpy(dBuff, hBuff, sizeWords * sizeof(T));
        }

        void syncNonBlockingD2H() {
            queue.memcpy(hBuff, dBuff, sizeWords * sizeof(T));
        }

        void syncNonBlockingH2D() {
            queue.memcpy(dBuff, hBuff, sizeWords * sizeof(T));
        }

        std::vector<T> toVector() {
            std::vector<T> vec;
            vec.resize(sizeWords);
            syncBlockingD2H();
            std::memcpy(vec.data(), hBuff, sizeWords * sizeof(T));
            return vec;
        }

        T*& getHostBuffer() {
            return hBuff;
        }

        const T* getHostBuffer() const {
            return hBuff;
        }

        T*& getDeviceBuffer() {
            return dBuff;
        }

        const T* getDeviceBuffer() const {
            return dBuff;
        }

        void saveHostToNpy(size_t offset, size_t lenWords, const std::string &npyFile);
    };


    template<typename TT>
    using TensorPtr = std::shared_ptr<Tensor<TT>>;

    void fillTensorWithRandomData(Tensor<float> &t);
    void fillTensorWith(Tensor<float> &t, float val);

    void fillTensorWithRandomData(Tensor<int> &t, int valUpperLimit);

}