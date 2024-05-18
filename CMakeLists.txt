cmake_minimum_required(VERSION 3.14)

# CONFIGURATION
set(LOG_LEVEL 0 CACHE STRING "0: Trace, 1: Debug, 2: Info, 3: Warn, 4: Error, 5: Critical, 6: OFF")


set(CMAKE_CXX_COMPILER icpx)
project(HPC_Course_2024 CXX CUDA)
add_compile_definitions(SPDLOG_FMT_EXTERNAL)
add_definitions(-DLOG=${LOG_LEVEL})
set(BASE_FLAGS
        $<$<CONFIG:Debug>:-Wall -Wextra -Wno-unused-parameter -fno-common -g -ggdb -Wno-unknown-pragmas -fsycl -fsycl-link -fsycl-targets=nvptx64-nvidia-cuda,spir64 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80>
        $<$<CONFIG:Release>:-Wall -Wextra -Wno-unused-parameter -fno-common -O3 -funroll-loops -Wno-unknown-pragmas -fsycl -fsycl-link -fsycl-targets=nvptx64-nvidia-cuda,spir64 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80>
)
set(BASE_LINKER_FLAGS
        $<$<CONFIG:Debug>:-fsycl>
        $<$<CONFIG:Release>:-fsycl>
)

find_package(GTest REQUIRED)
find_package(spdlog REQUIRED)
find_package(CUDAToolkit REQUIRED)
include(FetchContent)
FetchContent_Declare(
        libnpy
        URL https://github.com/llohse/libnpy/archive/refs/tags/v1.0.1.tar.gz
        URL_HASH MD5=0aba07020231be841d5eaa3e092c237b
)
FetchContent_MakeAvailable(libnpy libnpy)


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
