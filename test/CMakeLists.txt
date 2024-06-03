add_subdirectory(test-sycl)

add_executable(testAll
        testAll.cpp
        unittests/testCommon.h
        unittests/testTensor.cpp
        unittests/testKernelEncode.cpp
        #unittests/testKernelSoftmax.cpp
        unittests/testKernelLayerNorm.cpp
        #unittests/testKernelMatmulBias.cpp
        unittests/testOneMkl.cpp
)
target_compile_options(testAll PRIVATE ${BASE_FLAGS})
target_link_options(testAll PRIVATE ${BASE_LINKER_FLAGS})
target_link_libraries(
        testAll
        PRIVATE
        GTest::GTest
        spdlog::spdlog
        argparse
        common
        core
        kernels
        sycl
        onemkl
        onemkl_blas
        onemkl_dft
        onemkl_lapack
        onemkl_sparse_blas
        onemkl_rng
)
set_target_properties(testAll PROPERTIES NO_SYSTEM_FROM_IMPORTED true)
