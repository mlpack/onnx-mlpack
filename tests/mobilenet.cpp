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
using namespace mlpack;

TEST_CASE("test_mobilenet_subgraph_match", "[mobilenet]")
{
  const string onnxFilePath = "mobilenetv2-7.onnx";
  DAGNetwork<> generatedModel = onnx_mlpack::Convert(onnxFilePath);

  REQUIRE(generatedModel.SortedNetwork().size() > 0);

  // We could load an image with mlpack's data loader, but STB might choose a
  // slightly different pixel value than matplotlib's image loader, so we
  // instead just load images that were exported from matplotlib.
  arma::mat inputData;
  Load("mobilenet_inputs.csv", inputData, Fatal);

  arma::mat outputData;
  Load("mobilenet_outputs.csv", outputData, Fatal);

  // Test running some inputs through the network.
  arma::mat actualOutputs;
  generatedModel.Predict(inputData, actualOutputs);

  REQUIRE(actualOutputs.n_cols == outputData.n_cols);
  REQUIRE(actualOutputs.n_rows == outputData.n_rows);
  REQUIRE(approx_equal(actualOutputs, outputData, "both", 0.001, 0.001));
}
>>>>>>> origin/master
