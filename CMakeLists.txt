cmake_minimum_required(VERSION 3.14)


# CONFIGURATION vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
set(LOG_LEVEL 0 CACHE STRING "0: Trace, 1: Debug, 2: Info, 3: Warn, 4: Error, 5: Critical, 6: OFF")
set(USE_NVIDIA ON CACHE BOOL "Use NVIDIA GPU")
set(USE_NVIDIA_ARCH "sm_89" CACHE STRING "NVIDIA GPU architecture, only used if USE_NVIDIA is ON.")
set(USE_INTEL OFF CACHE BOOL "Use INTEL GPU")
set(USE_INTEL_ARCH "ffffffffff" CACHE STRING "INTEL GPU architecture, only used if USE_INTEL is ON.")
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


if (USE_NVIDIA AND USE_INTEL)
    message(FATAL_ERROR "Cannot use both NVIDIA and INTEL GPUs")
endif ()
set(CMAKE_CXX_COMPILER icpx) # Always use ICPX, even for CUDA

if (USE_NVIDIA)
    project(HPC_Course_2024 CXX CUDA)
else ()
    project(HPC_Course_2024 CXX)
endif ()
set(SYCL_AOT_FLAGS_COMPILE "NotDefined")
set(SYCL_AOT_FLAGS_LINK "NotDefined")
if (USE_NVIDIA)
    set(SYCL_AOT_FLAGS_COMPILE "-fsycl-targets=nvptx64-nvidia-cuda --cuda-gpu-arch=${USE_NVIDIA_ARCH}")
    set(SYCL_AOT_FLAGS_LINK "-fsycl-targets=nvptx64-nvidia-cuda")
endif ()
if (USE_INTEL)
    set(SYCL_AOT_FLAGS_COMPILE "-fsycl-targets=${USE_INTEL_ARCH}")
    set(SYCL_AOT_FLAGS_LINK "-fsycl-targets=${USE_INTEL_ARCH}")
endif ()


add_compile_definitions(SPDLOG_FMT_EXTERNAL)
add_definitions(-DLOG=${LOG_LEVEL})
set(BASE_FLAGS
        $<$<CONFIG:Debug>:-Wall -Wextra -Wno-unused-parameter -fno-common -g -ggdb -Wno-unknown-pragmas -fsycl -fsycl-link ${SYCL_AOT_FLAGS_COMPILE}>
        $<$<CONFIG:Release>:-Wall -Wextra -Wno-unused-parameter -fno-common -O3 -funroll-loops -Wno-unknown-pragmas -ffast-math -fsycl -fsycl-link ${SYCL_AOT_FLAGS_COMPILE}>
)
set(BASE_LINKER_FLAGS
        $<$<CONFIG:Debug>:-fsycl ${SYCL_AOT_FLAGS_LINK}> # -fsycl-device-code-split=per_kernel>
        $<$<CONFIG:Release>:-fsycl ${SYCL_AOT_FLAGS_LINK}> # -fsycl-device-code-split=per_kernel>
)

find_package(GTest REQUIRED)
find_package(spdlog REQUIRED)
if (USE_NVIDIA)
    find_package(CUDAToolkit REQUIRED)
endif ()


include(FetchContent)
FetchContent_Declare(
        libnpy
        URL https://github.com/llohse/libnpy/archive/refs/tags/v1.0.1.tar.gz
        URL_HASH MD5=0aba07020231be841d5eaa3e092c237b
)
FetchContent_MakeAvailable(libnpy libnpy)

set(BUILD_FUNCTIONAL_TESTS OFF)
set(BUILD_EXAMPLES OFF)
set(BUILD_EXAMPLES OFF)
if (USE_NVIDIA)
    set(ENABLE_CUBLAS_BACKEND ON)
    set(ENABLE_CUFFT_BACKEND ON)
    set(ENABLE_CURAND_BACKEND ON)
endif ()
set(ENABLE_OMP_OFFLOAD ON)
FetchContent_Declare(
        open_onemkl
        GIT_REPOSITORY https://github.com/oneapi-src/oneMKL.git
        GIT_TAG 028d8ba
)
FetchContent_MakeAvailable(open_onemkl)

# fetch latest argparse
include(FetchContent)
FetchContent_Declare(
        argparse
        GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(src)
add_subdirectory(test)


message(STATUS "C compiler used by CMake: ${CMAKE_C_COMPILER}")
message(STATUS "CXX compiler used by CMake: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA compiler used by CMake: ${CMAKE_CUDA_COMPILER}")
message(STATUS "Linker used by CMake: ${CMAKE_LINKER}")

