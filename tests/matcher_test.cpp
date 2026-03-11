/**
 * @file matcher_test.cpp
 *
 * Test that a simple network can be matched to an mlpack graph.
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
using namespace onnx_mlpack;

// Check that we can load a manually-crafted ONNX network consisting of three
// LinearNoBias layers.
TEST_CASE("test_manual_linearnobias_network", "[iris]")
{
  // Load the ONNX graph.
  const string onnxFilePath = "linear_no_bias.onnx";
  onnx::GraphProto graph = GetGraph(onnxFilePath);

  // Get the mlpack model from the graph.
  DAGNetwork<> generatedModel = SubgraphConvert(graph);

  REQUIRE(generatedModel.Network().size() == 3);

  vector<Layer<>*> sortedLayers = generatedModel.SortedNetwork();

  // Make sure all layers have the correct type.
  REQUIRE(sortedLayers.size() == 3);
  REQUIRE(dynamic_cast<LinearNoBias<>*>(sortedLayers[0]) != nullptr);
  REQUIRE(dynamic_cast<LinearNoBias<>*>(sortedLayers[1]) != nullptr);
  REQUIRE(dynamic_cast<LinearNoBias<>*>(sortedLayers[2]) != nullptr);
}
