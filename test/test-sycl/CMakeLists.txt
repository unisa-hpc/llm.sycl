

set(SOURCE_FILE TestVecAdd.cpp)
set(TARGET_NAME TestVecAdd)
add_executable(${TARGET_NAME} ${SOURCE_FILE})
target_compile_options(${TARGET_NAME} PRIVATE ${BASE_FLAGS})
target_link_options(${TARGET_NAME} PRIVATE ${BASE_LINKER_FLAGS})