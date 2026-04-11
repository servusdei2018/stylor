include(FetchContent)

FetchContent_Declare(
    cli11_proj
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.4.2
)
FetchContent_MakeAvailable(cli11_proj)

FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
)
FetchContent_MakeAvailable(json)

add_library(stylor STATIC
    stylor.cpp
    image.cpp
    tensor.cpp
    loss.cpp
    optimizer.cpp
    vgg/vgg.cpp
    vgg/preprocessing.cpp
    vgg/weight_loader.cpp
    safetensors_io.cpp
    transform_network.cpp
)

target_include_directories(stylor 
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${stb_SOURCE_DIR}
)

target_link_libraries(stylor 
    PUBLIC 
        dnnl
    PRIVATE 
        nlohmann_json::nlohmann_json
)

add_executable(stylor_cli 
    main.cpp
    commands/infer.cpp
    commands/train.cpp
)

target_link_libraries(stylor_cli 
    PRIVATE 
        stylor 
        CLI11::CLI11
)

set_target_properties(stylor_cli PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)