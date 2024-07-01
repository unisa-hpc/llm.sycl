add_library(layers "")
target_compile_options(layers PRIVATE ${BASE_FLAGS})
target_link_options(layers PRIVATE ${BASE_LINKER_FLAGS})

target_sources(layers
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/BaseLayer.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Attention.cpp
        ${CMAKE_CURRENT_LIST_DIR}/MatmulBias.cpp
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/BaseLayer.h
        ${CMAKE_CURRENT_LIST_DIR}/Attention.h
        ${CMAKE_CURRENT_LIST_DIR}/MatmulBias.h
)
target_include_directories(layers
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${libnpy_SOURCE_DIR}/include
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/..
)
target_link_libraries(layers
        PRIVATE
        common
        core
        kernels
        spdlog::spdlog
        sycl
        PUBLIC
        onemkl
        onemkl_blas
        onemkl_dft
        onemkl_lapack
        onemkl_sparse_blas
        onemkl_rng
)