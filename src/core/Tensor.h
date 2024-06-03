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
        /**
         * @brief Construct a new Tensor object
         * @note Does not sync anything from host to the device, as it is pointless!
         * @param queue
         * @param shape
         */
        Tensor(sycl::queue &queue, const std::vector<size_t> &shape);

        /**
         * @brief Construct a new Tensor object
         * @note In all configurations, the final data in the new Tensor will be synced between the host and the device.
         * @param queue
         * @param shape
         * @param vecData
         */
        Tensor(sycl::queue &queue, std::vector<size_t> shape, std::vector<T> vecData);

        /**
         * @brief Construct a new Tensor object
         * @note In all configurations, the final data in the new Tensor will be synced between the host and the device.
         * @param queue
         * @param shape
         * @param buff
         */
        Tensor(sycl::queue &queue, std::vector<size_t> shape, const T *buff);

        /**
         * @brief Construct a new Tensor object
         * @warning Uses the queue stored in `other`.
         * @note In all configurations, the final data in the new Tensor will be synced between the host and the device.
         * @param other The other tensor to be cloned.
         * @param fromItsDeviceBuffer When true, the device buffer of `other` will be copied to the new Tensor. Otherwise, the host buffer will be copied.
         */
        Tensor(Tensor &other, bool fromItsDeviceBuffer = false);

        /**
         * @brief Destroy the Tensor object and releases the host and the device buffers (not sycl::buffer).
         */
        ~Tensor();

        /**
         * @brief Save the host buffer to a npy file.
         * @warning Be aware that this method has nothing to do with the device buffer. It does not sync anything either.
         * @param npyFile
         */
        void saveHostToNpy(const std::string &npyFile);

        /**
         * @brief Get the shape of the Tensor.
         * @return A cloned vector of the shape.
         */
        std::vector<size_t> getShape() const;

        /**
         * @brief Get the shape of the Tensor as a string.
         * @return A string representing the shape.
         */
        std::string getShapeStr() const;

        /**
         * @brief Get the size of the Tensor in words.
         * @return The size of the Tensor in words (not bytes!).
         */
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

        /**
         * @brief Load a npy file to the host buffer.
         * @note The device buffer **will be** synced with the host buffer.
         * @param queue
         * @param npyFile
         * @return A pointer to the new Tensor.
         */
        static Tensor<T> loadToHost(sycl::queue &queue, std::string npyFile);

        /**
         * @brief Sync the device buffer to the host buffer in a blocking manner.
         */
        void syncBlockingD2H() {
            queue.memcpy(hBuff, dBuff, sizeWords * sizeof(T));
            queue.wait_and_throw();
        }

        /**
         * @brief Sync the host buffer to the device buffer in a blocking manner.
         */
        void syncBlockingH2D() {
            queue.memcpy(dBuff, hBuff, sizeWords * sizeof(T));
            queue.wait_and_throw();
        }

        /**
         * @brief Sync the device buffer to the host buffer in a non-blocking manner.
         */
        void syncNonBlockingD2H() {
            queue.memcpy(hBuff, dBuff, sizeWords * sizeof(T));
        }

        /**
         * @brief Sync the host buffer to the device buffer in a non-blocking manner.
         */
        void syncNonBlockingH2D() {
            queue.memcpy(dBuff, hBuff, sizeWords * sizeof(T));
        }

        /**
         * @brief Convert the host buffer to a vector.
         * @note This method has nothing to do with the device buffer. It does not sync anything either.
         * @return A vector of the data.
         */
        std::vector<T> toVectorHostOnly() {
            std::vector<T> vec;
            vec.resize(sizeWords);
            std::memcpy(vec.data(), hBuff, sizeWords * sizeof(T));
            return vec;
        }

        /**
         * @brief Get a reference pointer to the host buffer.
         * @return A reference pointer to the host buffer.
         */
        T*& getHostBuffer() {
            return hBuff;
        }
        /**
         * @brief Get a const pointer to the host buffer pointer.
         * @return A const pointer to the host buffer.
         */
        const T* getHostBuffer() const {
            return hBuff;
        }

        /**
         * @brief Get a reference pointer to the device buffer.
         * @return A reference pointer to the device buffer.
         */
        T*& getDeviceBuffer() {
            return dBuff;
        }

        /**
         * @brief Get a const pointer to the device buffer.
         * @return A const pointer to the device buffer.
         */
        const T* getDeviceBuffer() const {
            return dBuff;
        }

        /**
         * @brief Save a portion of the host buffer to a npy file.
         * @warning Be aware that this method has nothing to do with the device buffer. It does not sync anything either.
         * @param offset The offset in words.
         * @param lenWords The length in words.
         * @param npyFile The path to the npy file to be created.
         */
        void saveHostToNpy(size_t offset, size_t lenWords, const std::string &npyFile);
    };


    template<typename TT>
    using TensorPtr = std::shared_ptr<Tensor<TT>>;

    /**
     * @brief Fill a tensor with random data.
     * @note The data will be synced with the device.
     * @param t The tensor to be filled.
     */
    void fillTensorWithRandomData(Tensor<float> &t);

    /**
     * @brief Fill a tensor with a specific value.
     * @param t
     * @param val
     */
    void fillTensorWith(Tensor<float> &t, float val);

    /**
     * @brief Fill a tensor with random integer data with a upper limit on the integers.
     * @param t
     * @param valUpperLimit
     */
    void fillTensorWithRandomData(Tensor<int> &t, int valUpperLimit);

}