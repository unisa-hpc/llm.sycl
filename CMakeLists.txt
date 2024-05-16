cmake_minimum_required(VERSION 3.14)
if (UNIX)
    # Direct CMake to use icpx rather than the default C++ compiler/linker
    set(CMAKE_CXX_COMPILER icpx)
else () # Windows
    # Force CMake to use icx-cl rather than the default C++ compiler/linker
    # (needed on Windows only)
    include(CMakeForceCompiler)
    CMAKE_FORCE_CXX_COMPILER(icx-cl IntelDPCPP)
    include(Platform/Windows-Clang)
endif ()

project(HPC_Course_2024 CXX CUDA)



# CONFIGURATION
set(LOG_LEVEL 0 CACHE STRING "0: Trace, 1: Debug, 2: Info, 3: Warn, 4: Error, 5: Critical, 6: OFF")




add_compile_definitions(SPDLOG_FMT_EXTERNAL)
add_definitions(-DLOG=${LOG_LEVEL})
set(BASE_FLAGS
        $<$<CONFIG:Debug>:-Wall -Wextra -Wno-unused-parameter -fno-common -g -ggdb -Wno-unknown-pragmas>
        $<$<CONFIG:Release>:-Wall -Wextra -Wno-unused-parameter -fno-common -O3 -funroll-loops -Wno-unknown-pragmas>
)

find_package(GTest REQUIRED)
find_package(spdlog REQUIRED)
find_package(CUDAToolkit REQUIRED)

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
