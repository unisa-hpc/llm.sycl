#pragma once

#include <string>
#include <map>
#include <sycl/sycl.hpp>
#include "common/common.h"
#include "core/Tensor.h"

namespace llmsycl::layers {
    class BaseLayer {
    protected:
        std::string layerName;
        std::vector<std::string> reportParamsTensors;
        std::vector<std::string> reportParamsScalars;
    public:
        BaseLayer(const std::string &layerName) :
                layerName(layerName) {
        }

        virtual std::vector<sycl::event> Launch(sycl::queue &q, const std::vector<sycl::event> &dependencies)=0;

        size_t LaunchBlockingAndMeasureNanoSec(sycl::queue &q, const std::vector<sycl::event> &dependencies) {
            size_t sumElapsed = 0;
            auto eventsVec = Launch(q, dependencies);
            for (auto &e : eventsVec)
                e.wait();
            for (auto &e : eventsVec) {
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
                logger->info("Layer Queued: {}", layerName);
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


