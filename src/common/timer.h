#pragma once

#include "common.h"

#include <chrono>
#include <functional>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

class Timer {
private:
    std::chrono::system_clock::time_point m_oTimerLast;
public:
    Timer() {
        m_oTimerLast = high_resolution_clock::now();
    }

    template<class StdTimeResolution = std::milli>
    float FromLast() {
        auto now = high_resolution_clock::now();
        duration<float, StdTimeResolution> ms = now - m_oTimerLast;
        m_oTimerLast = now;
        return ms.count();
    }

    template<class StdTimeResolution = std::milli>
    float ReportFromLast(const std::string &msg="") {
        auto t = FromLast<StdTimeResolution>();
        logger->trace("Elapsed {}: {} .", msg, t);
        return t;
    }

    template<class StdTimeResolution = std::milli>
    static inline float ForLambda(const std::function<void()> &operation) {
        auto t1 = high_resolution_clock::now();
        operation();
        auto t2 = high_resolution_clock::now();
        duration<float, StdTimeResolution> ms = t2 - t1;
        return ms.count();
    }

    template<class StdTimeResolution = std::milli>
    static inline float ReportForLambda(const std::function<void()> &operation) {
        auto t = ForLambda<StdTimeResolution>(operation);
        logger->trace("Elapsed: {} .", t);
        return t;
    }
};