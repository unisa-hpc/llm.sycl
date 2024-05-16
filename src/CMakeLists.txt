add_subdirectory(common)
add_subdirectory(orig)

add_executable(LLM_SYCL main.cpp)
target_compile_options(LLM_SYCL PRIVATE ${BASE_FLAGS})
target_link_libraries(LLM_SYCL argparse common)