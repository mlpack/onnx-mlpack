/**
 * @file yolo-tiny.cpp
 *
 * Test that the YOLOv3-Tiny network can be loaded from ONNX correctly.
 */
#include <onnx_mlpack.hpp>
#include "catch.hpp"

using namespace std;
using namespace mlpack;

TEST_CASE("tiny_yolo_subgraph_match", "[yolo-tiny]")
{
  // Load the ONNX network.
  const string onnxFilePath = "tinyyolo-v2.3-o8.onnx";
  onnx::GraphProto graph = onnx_mlpack::GetGraph(onnxFilePath);

  // Convert to mlpack DAGNetwork.
  DAGNetwork<> generatedModel = onnx_mlpack::Convert(graph);

  REQUIRE(generatedModel.Network().size() == 38);

  // Make sure we extracted the right structure of the network.
  REQUIRE(dynamic_cast<Scale<>*>(generatedModel.SortedNetwork()[0]) != nullptr);
  REQUIRE(dynamic_cast<Convolution<>*>(generatedModel.SortedNetwork()[1]) !=
      nullptr);
  REQUIRE(dynamic_cast<BatchNorm<>*>(generatedModel.SortedNetwork()[2]) !=
      nullptr);
  REQUIRE(dynamic_cast<LeakyReLU<>*>(generatedModel.SortedNetwork()[3]) !=
      nullptr);
  REQUIRE(dynamic_cast<Padding<>*>(generatedModel.SortedNetwork()[4]) !=
      nullptr);
  REQUIRE(dynamic_cast<MaxPooling<>*>(generatedModel.SortedNetwork()[5]) !=
      nullptr);
  REQUIRE(dynamic_cast<Convolution<>*>(generatedModel.SortedNetwork()[6]) !=
      nullptr);
  REQUIRE(dynamic_cast<BatchNorm<>*>(generatedModel.SortedNetwork()[7]) !=
      nullptr);
  REQUIRE(dynamic_cast<LeakyReLU<>*>(generatedModel.SortedNetwork()[8]) !=
      nullptr);
  REQUIRE(dynamic_cast<Padding<>*>(generatedModel.SortedNetwork()[9]) !=
      nullptr);
  REQUIRE(dynamic_cast<MaxPooling<>*>(generatedModel.SortedNetwork()[10]) !=
      nullptr);
  REQUIRE(dynamic_cast<Convolution<>*>(generatedModel.SortedNetwork()[11]) !=
      nullptr);
  REQUIRE(dynamic_cast<BatchNorm<>*>(generatedModel.SortedNetwork()[12]) !=
      nullptr);
  REQUIRE(dynamic_cast<LeakyReLU<>*>(generatedModel.SortedNetwork()[13]) !=
      nullptr);
  REQUIRE(dynamic_cast<Padding<>*>(generatedModel.SortedNetwork()[14]) !=
      nullptr);
  REQUIRE(dynamic_cast<MaxPooling<>*>(generatedModel.SortedNetwork()[15]) !=
      nullptr);
  REQUIRE(dynamic_cast<Convolution<>*>(generatedModel.SortedNetwork()[16]) !=
      nullptr);
  REQUIRE(dynamic_cast<BatchNorm<>*>(generatedModel.SortedNetwork()[17]) !=
      nullptr);
  REQUIRE(dynamic_cast<LeakyReLU<>*>(generatedModel.SortedNetwork()[18]) !=
      nullptr);
  REQUIRE(dynamic_cast<Padding<>*>(generatedModel.SortedNetwork()[19]) !=
      nullptr);
  REQUIRE(dynamic_cast<MaxPooling<>*>(generatedModel.SortedNetwork()[20]) !=
      nullptr);
  REQUIRE(dynamic_cast<Convolution<>*>(generatedModel.SortedNetwork()[21]) !=
      nullptr);
  REQUIRE(dynamic_cast<BatchNorm<>*>(generatedModel.SortedNetwork()[22]) !=
      nullptr);
  REQUIRE(dynamic_cast<LeakyReLU<>*>(generatedModel.SortedNetwork()[23]) !=
      nullptr);
  REQUIRE(dynamic_cast<Padding<>*>(generatedModel.SortedNetwork()[24]) !=
      nullptr);
  REQUIRE(dynamic_cast<MaxPooling<>*>(generatedModel.SortedNetwork()[25]) !=
      nullptr);
  REQUIRE(dynamic_cast<Convolution<>*>(generatedModel.SortedNetwork()[26]) !=
      nullptr);
  REQUIRE(dynamic_cast<BatchNorm<>*>(generatedModel.SortedNetwork()[27]) !=
      nullptr);
  REQUIRE(dynamic_cast<LeakyReLU<>*>(generatedModel.SortedNetwork()[28]) !=
      nullptr);
  REQUIRE(dynamic_cast<Padding<>*>(generatedModel.SortedNetwork()[29]) !=
      nullptr);
  REQUIRE(dynamic_cast<MaxPooling<>*>(generatedModel.SortedNetwork()[30]) !=
      nullptr);
  REQUIRE(dynamic_cast<Convolution<>*>(generatedModel.SortedNetwork()[31]) !=
      nullptr);
  REQUIRE(dynamic_cast<BatchNorm<>*>(generatedModel.SortedNetwork()[32]) !=
      nullptr);
  REQUIRE(dynamic_cast<LeakyReLU<>*>(generatedModel.SortedNetwork()[33]) !=
      nullptr);
  REQUIRE(dynamic_cast<Convolution<>*>(generatedModel.SortedNetwork()[34]) !=
      nullptr);
  REQUIRE(dynamic_cast<BatchNorm<>*>(generatedModel.SortedNetwork()[35]) !=
      nullptr);
  REQUIRE(dynamic_cast<LeakyReLU<>*>(generatedModel.SortedNetwork()[36]) !=
      nullptr);
  REQUIRE(dynamic_cast<Convolution<>*>(generatedModel.SortedNetwork()[37]) !=
      nullptr);

  // We could load an image with mlpack's data loader, but STB might choose a
  // slightly different pixel value than matplotlib's image loader, so we
  // instead just load images that were exported from matplotlib.
  arma::mat inputData;
  Load("tinyyolo_inputs.csv", inputData, Fatal);

  arma::mat outputData;
  Load("tinyyolo_outputs.csv", outputData, Fatal);

  // Test running some inputs through the network.
  arma::mat actualOutputs;
  generatedModel.Predict(inputData, actualOutputs);

  REQUIRE(actualOutputs.n_cols == outputData.n_cols);
  REQUIRE(actualOutputs.n_rows == outputData.n_rows);
  REQUIRE(approx_equal(actualOutputs, outputData, "both", 0.001, 0.001));
}
