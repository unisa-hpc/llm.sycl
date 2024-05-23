add_library(kernels "")
target_compile_options(kernels PRIVATE ${BASE_FLAGS})
target_link_options(kernels PRIVATE ${BASE_LINKER_FLAGS})

target_sources(kernels
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/Helpers.cpp
        ${CMAKE_CURRENT_LIST_DIR}/BaseKernel.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Encoder.cpp
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/Helpers.h
        ${CMAKE_CURRENT_LIST_DIR}/BaseKernel.h
        ${CMAKE_CURRENT_LIST_DIR}/Encoder.h
)
target_include_directories(kernels
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${libnpy_SOURCE_DIR}/include
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/..
)
target_link_libraries(kernels
        PRIVATE
        common
        core
        spdlog::spdlog
        sycl
)