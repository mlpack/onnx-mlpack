/**
 * @file iris.cpp
 *
 * Test that a simple network for iris dataset classification can be loaded from
 * ONNX correctly.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <onnx_mlpack.hpp>
#include "catch.hpp"

using namespace std;
using namespace mlpack;

// Check that we can load the ONNX iris network, and check the network
// structure.
TEST_CASE("test_iris_onnx_load", "[iris]")
{
  // Load the ONNX graph.
  const string onnxFilePath = "iris_model.onnx";
  onnx::GraphProto graph = onnx_mlpack::GetGraph(onnxFilePath);

  // Get the mlpack model from the graph.
  DAGNetwork<> generatedModel = onnx_mlpack::Convert(graph);

  REQUIRE(generatedModel.Network().size() == 4);

  vector<Layer<>*> sortedLayers = generatedModel.SortedNetwork();

  // Make sure all layers have the correct type.
  REQUIRE(sortedLayers.size() == 4);
  REQUIRE(dynamic_cast<Linear<>*>(sortedLayers[0]) != nullptr);
  REQUIRE(dynamic_cast<ReLU<>*>(sortedLayers[1]) != nullptr);
  REQUIRE(dynamic_cast<Linear<>*>(sortedLayers[2]) != nullptr);
  REQUIRE(dynamic_cast<Softmax<>*>(sortedLayers[3]) != nullptr);
}

TEST_CASE("test_iris_convert_and_predict", "[iris]")
{
  // Load the ONNX graph.
  const string onnxFilePath = "iris_model.onnx";
  onnx::GraphProto graph = onnx_mlpack::GetGraph(onnxFilePath);

  // Get the mlpack model from the graph.
  DAGNetwork<> generatedModel = onnx_mlpack::Convert(graph);

  // Load the iris data for classification.
  arma::mat data;
  REQUIRE(Load("iris.csv", data) == true);
  arma::urowvec labels;
  REQUIRE(Load("iris_labels.txt", labels) == true);

  // Normalize the features.
  MinMaxScaler scaler;
  scaler.Fit(data);
  scaler.Transform(data, data);

  // Make the predictions.
  arma::mat predictions;
  generatedModel.Predict(data, predictions);

  arma::urowvec predictedLabels = index_max(predictions, 0);

  REQUIRE(labels.n_elem == predictedLabels.n_elem);
  REQUIRE(double(accu(labels == predictedLabels)) / labels.n_elem >= 0.9);
}
