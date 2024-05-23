//
// Created by saleh on 23/05/24.
//

#pragma once

#include <string>
#include "common/common.h"
#include "core/Tensor.h"

namespace llmsycl::kernels {
    class Helpers {
        public:
        static constexpr int CeilDiv(int M, int N) {
            return (M + N - 1) / N;
        }

        static constexpr int MakeDivisible(int val, int to) {
            return CeilDiv(val, to) * to;
        }

    };

}
