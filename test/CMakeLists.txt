add_subdirectory(test-sycl)

add_executable(testAll
        testAll.cpp
        unittests/testTensor.cpp
        unittests/testKernelEncode.cpp
)
target_compile_options(testAll PRIVATE ${BASE_FLAGS})
target_link_options(testAll PRIVATE ${BASE_LINKER_FLAGS})
target_link_libraries(
        testAll
        PRIVATE
        GTest::GTest
        argparse
        common
        core
        kernels
        sycl
)