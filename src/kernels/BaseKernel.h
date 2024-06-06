//
// Created by saleh on 23/05/24.
//

#pragma once

#include <string>
#include "Helpers.h"
#include "common/common.h"
#include "spdlog/fmt/fmt.h"


namespace llmsycl::kernels {

    class BaseKernel {
    protected:
        const std::string kernelName;
        std::vector<std::string> reportParamsTensors;
        std::vector<std::string> reportParamsScalars;
    public:
        BaseKernel(const std::string &kernelName) :
                kernelName(kernelName) {
        }

        virtual std::vector<sycl::event> Launch(sycl::queue &q, int blockSize, const std::vector<sycl::event> &dependencies)=0;

        size_t LaunchBlockingAndMeasureNanoSec(sycl::queue &q, int blockSize, const std::vector<sycl::event> &dependencies) {
            size_t sumElapsed = 0;
            auto vecEvents = Launch(q, blockSize, dependencies);
            for(auto &e: vecEvents){
                e.wait();
            }
            for(auto &e: vecEvents){
                auto start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
                auto end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
                sumElapsed += end - start;
            }

            return sumElapsed;
        }

    protected:
        template<typename T>
        void addTensorDetailsToReport(const std::string &tensorName, const core::Tensor<T> &tensor) {
            if (!globalIsSilent) {
                reportParamsTensors.push_back(
                        fmt::format("Tensor: {} - Shape: {} - Size (words): {}",
                                    tensorName,
                                    tensor.getShapeStr(),
                                    tensor.getSize()
                        )
                );
            }
        }

        template<typename T>
        void addScalarParamToReport(const std::string &paramName, const T &paramValue) {
            if (!globalIsSilent) {
                reportParamsScalars.push_back(
                        fmt::format("Scalar Param: {} - Value: {}",
                                    paramName,
                                    paramValue
                        )
                );
            }
        }

        void report() {
            if (!globalIsSilent) {
                logger->info("Kernel Queued: {}", kernelName);
                for (const auto &line: reportParamsTensors) {
                    logger->debug("\t\t{}", line);
                }
                for (const auto &line: reportParamsScalars) {
                    logger->trace("\t\t{}", line);
                }
            }
        }
    };
}

