cmake_minimum_required(VERSION 3.4...3.22)

project(falcon_cpp)

option(FORCE_CMAKE "Force CMake build of Python bindings" OFF)

set(FORCE_CMAKE $ENV{FORCE_CMAKE})

if (UNIX AND NOT FORCE_CMAKE)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/vendor/ggllm.cpp/libllama.so
        COMMAND make libllama.so
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/vendor/ggllm.cpp
    )
    add_custom_target(
        run ALL
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/vendor/ggllm.cpp/libllama.so
    )
    install(
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/vendor/ggllm.cpp/libllama.so
        DESTINATION llama_cpp
    )
else()
    set(BUILD_SHARED_LIBS "On")
    add_subdirectory(vendor/ggllm.cpp)
    install(
        TARGETS ggllm
        LIBRARY DESTINATION falcon_cpp
        RUNTIME DESTINATION falcon_cpp
        ARCHIVE DESTINATION falcon_cpp
        FRAMEWORK DESTINATION falcon_cpp
        RESOURCE DESTINATION falcon_cpp
    )
endif()
