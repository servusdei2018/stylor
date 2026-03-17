cmake_minimum_required(VERSION 3.20)
project(Stylor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

cmake_policy(SET CMP0135 NEW)

include_directories(${CMAKE_SOURCE_DIR}/include)

include(FetchContent)

FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
)
FetchContent_MakeAvailable(stb)

FetchContent_Declare(
    onednn
    GIT_REPOSITORY https://github.com/oneapi-src/oneDNN.git
    GIT_TAG v3.11.1
)
set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(DNNL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(onednn)

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)


find_program(CLANG_FORMAT NAMES clang-format)

if(CLANG_FORMAT)
    file(GLOB_RECURSE SOURCE_FILES 
        ${CMAKE_SOURCE_DIR}/src/*.cpp 
        ${CMAKE_SOURCE_DIR}/include/*.hpp 
        ${CMAKE_SOURCE_DIR}/tests/*.cpp 
        ${CMAKE_SOURCE_DIR}/tests/*.hpp 
        ${CMAKE_SOURCE_DIR}/*.cpp 
        ${CMAKE_SOURCE_DIR}/*.hpp 
    )

    add_custom_target(
        format
        COMMAND ${CLANG_FORMAT} -i --style=file ${SOURCE_FILES}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Formatting code with clang-format"
    )
else()
    message(WARNING "clang-format not found. Code formatting will not be available.")
endif()
