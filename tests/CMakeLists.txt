# The CMake documentation says this is a bad idea, and I don't care.
# Your build system should have a way to just include all the files in a directory as build dependencies.
# Note that my don't-care attitude here means that if you add a new test file, you really need to reconfigure with CMake altogether.
file(GLOB TEST_SOURCES "*.cpp")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_executable(onnx_mlpack_test ${TEST_SOURCES})
target_precompile_headers(onnx_mlpack_test PUBLIC
    <armadillo> <mlpack.hpp> <ensmallen.hpp>)
target_link_libraries(onnx_mlpack_test PRIVATE ${MLPACK_LIBRARIES} ${ONNX_LIBRARIES})

# Copy test models to build directory.
add_custom_command(TARGET onnx_mlpack_test
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/test_models/ ${PROJECT_BINARY_DIR})

# Copy test data to build directory.
add_custom_command(TARGET onnx_mlpack_test
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/test_data/ ${PROJECT_BINARY_DIR})

add_test(NAME onnx_mlpack_test COMMAND onnx_mlpack_test)
