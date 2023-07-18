cmake_minimum_required(VERSION 3.4...3.22)

project(falcon_cpp)

# Build shared libraries using custom command
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/vendor/ggllm.cpp/libllama.so
    COMMAND cmake -DLLAMA_CUBLAS=1 -DCUDAToolkit_ROOT=/usr/local/cuda/ -DBUILD_SHARED_LIBS=on ${CMAKE_CURRENT_SOURCE_DIR}/vendor/ggllm.cpp
    COMMAND make
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/vendor/ggllm.cpp
)
add_custom_target(
    build_shared_libs ALL
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/vendor/ggllm.cpp/libllama.so
)
# Install shared libraries
install(
    FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/vendor/ggllm.cpp/libcmpnct_unicode.so
        ${CMAKE_CURRENT_SOURCE_DIR}/vendor/ggllm.cpp/libggml_shared.so
        ${CMAKE_CURRENT_SOURCE_DIR}/vendor/ggllm.cpp/libfalcon.so
        ${CMAKE_CURRENT_SOURCE_DIR}/vendor/ggllm.cpp/libllama.so
    DESTINATION falcon_cpp
)
