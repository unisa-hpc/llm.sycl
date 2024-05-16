set(COMPILE_FLAGS "-fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 -Wall")
set(LINK_FLAGS "-fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80")


set(SOURCE_FILE TestVecAdd.cpp
        ../../src/main.cpp)
set(TARGET_NAME TestVecAdd)

# To compile in a single command:
#    icpx -fsycl <file>.cpp -o <file>
# CMake executes:
#    [compile] icpx -fsycl -o <file>.cpp.o -c <file>.cpp
#    [link]    icpx -fsycl <file>.cpp.o -o <file>
add_executable(${TARGET_NAME} ${SOURCE_FILE})
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")
set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
add_custom_target(cpu-gpu DEPENDS ${TARGET_NAME})