add_library(common "")
target_compile_options(common PRIVATE ${BASE_FLAGS})

target_sources(common
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/common.cpp
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/common.h
)
target_include_directories(common
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/..
)
target_link_libraries(common
        PRIVATE
        spdlog::spdlog
)