include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.16.0.zip
)
FetchContent_MakeAvailable(googletest)

add_executable(test_stylor 
    test_main.cpp
    test_image.cpp
    test_tensor.cpp
    test_vgg.cpp
    test_preprocessing.cpp
    test_weight_loader.cpp
)
target_link_libraries(test_stylor PRIVATE stylor gtest gtest_main dnnl)
set_target_properties(test_stylor PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_test(NAME StylorTest COMMAND $<TARGET_FILE:test_stylor>)
