add_executable(OrigTrain train_gpt2_fp32.cu)
#target_compile_options(OrigTrain PRIVATE ${BASE_FLAGS})

target_compile_features(OrigTrain PRIVATE cxx_std_17)
set_target_properties(OrigTrain PROPERTIES CXX_STANDARD_REQUIRED ON)
set_property(TARGET OrigTrain PROPERTY CUDA_ARCHITECTURES 89)
target_include_directories(OrigTrain PRIVATE ${libnpy_SOURCE_DIR}/include)

target_link_libraries(OrigTrain CUDA::cudart CUDA::cublas CUDA::cublasLt)







add_executable(OrigTest test_gpt2_fp32.cu buf_to_npy.h)
#target_compile_options(OrigTest PRIVATE ${BASE_FLAGS})

target_compile_features(OrigTest PRIVATE cxx_std_17)
set_target_properties(OrigTest PROPERTIES CXX_STANDARD_REQUIRED ON)
set_property(TARGET OrigTest PROPERTY CUDA_ARCHITECTURES 89)
target_include_directories(OrigTest PRIVATE ${libnpy_SOURCE_DIR}/include)

target_link_libraries(OrigTest CUDA::cudart CUDA::cublas CUDA::cublasLt)