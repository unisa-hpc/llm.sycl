add_library(core "")
target_compile_options(core PRIVATE ${BASE_FLAGS})
target_link_options(core PRIVATE ${BASE_LINKER_FLAGS})

target_sources(core
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/Tensor.cpp
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/Tensor.h
)
target_include_directories(core
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${libnpy_SOURCE_DIR}/include
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/..
)
target_link_libraries(core
        PRIVATE
        common
        spdlog::spdlog
)