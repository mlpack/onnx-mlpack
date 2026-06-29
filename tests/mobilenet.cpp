/**
 * @file mobilenet.cpp
 *
 * Test that MobileNet can be loaded from ONNX correctly.
 * Information about the onnx mobilenet v7 model can be found at:
 * https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet
 */
#include <onnx_mlpack.hpp>
#include "catch.hpp"

using namespace std;
using namespace onnx_mlpack;

