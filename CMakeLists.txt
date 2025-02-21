cmake_minimum_required(VERSION 3.20)
project(Stylor LANGUAGES CXX)

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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

cmake_policy(SET CMP0135 NEW)

add_subdirectory(src)
add_subdirectory(tests)

enable_testing()
