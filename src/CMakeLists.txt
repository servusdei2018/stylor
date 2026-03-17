include(FetchContent)

add_library(stylor STATIC
    stylor.cpp
    image.cpp
    tensor.cpp
    vgg/vgg.cpp
    vgg/preprocessing.cpp
    vgg/weight_loader.cpp
)

target_include_directories(stylor PUBLIC
    ${stb_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(stylor PUBLIC dnnl)

FetchContent_Declare(
    cli11_proj
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.4.2
)
FetchContent_MakeAvailable(cli11_proj)

file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)

add_executable(stylor_cli ${SOURCES})
target_link_libraries(stylor_cli PRIVATE stylor CLI11::CLI11)
set_target_properties(stylor_cli PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)