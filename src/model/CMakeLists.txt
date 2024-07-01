add_library(model "")
target_compile_options(model PRIVATE ${BASE_FLAGS})
target_link_options(model PRIVATE ${BASE_LINKER_FLAGS})

target_sources(model
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/gpt2_v1/Model.cpp
        #${CMAKE_CURRENT_LIST_DIR}/gpt2_v1/DataLoader.cpp
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/gpt2_v1/Model.h
        #${CMAKE_CURRENT_LIST_DIR}/gpt2_v1/DataLoader.h
)
target_include_directories(model
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${libnpy_SOURCE_DIR}/include
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/..
)
target_link_libraries(model
        PRIVATE
        core
        common
        kernels
        layers
        spdlog::spdlog
)