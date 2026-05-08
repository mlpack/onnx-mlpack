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
TEST_CASE("test_manual_linearnobias_network", "[matching]")
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

// Check that we can translate a manually-crafted ONNX network, and that when we
// run it, we get pretty much the same thing.
TEST_CASE("test_manual_linearnobias_network_forward", "[matching]")
{
  // Load the ONNX graph.
  const string onnxFilePath = "linear_no_bias.onnx";
  onnx::GraphProto graph = GetGraph(onnxFilePath);
  DAGNetwork<> generatedModel = SubgraphConvert(graph);

  // These are the results from running this graph with the Python onnxruntime
  // package.  (3 points.)
  arma::mat inputs({{  0.5390724,  -0.1012873,   0.05991365 },
                    { -0.40542722, -0.04048201,  1.9344587  },
                    {  1.5806129,   1.197857,    1.3219868  },
                    {  0.7133332,   1.4374087,  -0.47156942 },
                    {  0.7104756,   1.2901834,   0.3671152  },
                    {  1.077111,    0.19215032,  0.5961245  },
                    {  0.84191304,  0.03732246, -1.3242655  },
                    {  0.7427492,   1.3101844,  -0.9769988  },
                    { -0.33089197, -0.33193994, -0.3888457  },
                    {  0.88787556,  1.1002026,   0.16943762 },
                    {  0.47336888,  0.5108939,  -0.0131184  },
                    { -0.4136899,   1.2215924,  -0.08011232 }});

  arma::mat outputs({{   6.781561,  -66.273964,    11.977367   },
                     { -13.906016,  -18.252462,    22.651491   },
                     { -10.938393,   39.402992,   -22.501211   },
                     { -22.654623,   26.099024,    11.084932   },
                     { -11.387621,  -34.205936,   -13.301408   },
                     {  -0.6678234, -25.290014,    -6.027281   },
                     {  -2.4156294,  19.373713,   -21.026665   },
                     {  -3.7015562,  -3.6036043,   -0.12286687 },
                     {  -7.058354,   -0.68583715,  19.433565   },
                     {   2.0760043, -29.713842,   -11.144423   }});

  arma::mat actualOutputs;
  generatedModel.Predict(inputs, actualOutputs);

  // Make sure that predictions reasonably match what the ONNX runtime produced.
  REQUIRE(actualOutputs.n_cols == outputs.n_cols);
  REQUIRE(actualOutputs.n_rows == outputs.n_rows);
  REQUIRE(approx_equal(outputs, actualOutputs, "both", 0.001, 0.001));
}

TEST_CASE("test_pytorch_linear_no_bias", "[matching]")
{
  // Load the ONNX graph.
  onnx::GraphProto graph = GetGraph("pytorch_linear_no_bias.onnx");
  DAGNetwork<> dag = SubgraphConvert(graph);

  REQUIRE(dag.Network().size() == 3);

  vector<Layer<>*> sortedLayers = dag.SortedNetwork();

  REQUIRE(sortedLayers.size() == 3);
  REQUIRE(dynamic_cast<LinearNoBias<>*>(sortedLayers[0]) != nullptr);
  REQUIRE(sortedLayers[0]->OutputDimensions().size() == 1);
  REQUIRE(sortedLayers[0]->OutputDimensions()[0] == 25);
  REQUIRE(dynamic_cast<LinearNoBias<>*>(sortedLayers[1]) != nullptr);
  REQUIRE(sortedLayers[1]->OutputDimensions().size() == 1);
  REQUIRE(sortedLayers[1]->OutputDimensions()[0] == 10);
  REQUIRE(dynamic_cast<LinearNoBias<>*>(sortedLayers[2]) != nullptr);
  REQUIRE(sortedLayers[2]->OutputDimensions().size() == 1);
  REQUIRE(sortedLayers[2]->OutputDimensions()[0] == 3);

  arma::mat input, outputRef;
  mlpack::Load("pytorch_linear_no_bias_inputs.csv", input, Fatal);
  mlpack::Load("pytorch_linear_no_bias_outputs.csv", outputRef, Fatal);

  arma::mat output;
  dag.Predict(input, output);

  REQUIRE(approx_equal(output, outputRef, "both", 0.001, 0.001));
}

TEST_CASE("test_pytorch_linear_no_bias_dynamo", "[matching]")
{
  // Load the ONNX graph.
  onnx::GraphProto graph = GetGraph("pytorch_linear_no_bias_dynamo.onnx");
  DAGNetwork<> dag = SubgraphConvert(graph);

  REQUIRE(dag.Network().size() == 3);

  vector<Layer<>*> sortedLayers = dag.SortedNetwork();

  REQUIRE(sortedLayers.size() == 3);
  REQUIRE(dynamic_cast<LinearNoBias<>*>(sortedLayers[0]) != nullptr);
  REQUIRE(sortedLayers[0]->OutputDimensions().size() == 1);
  REQUIRE(sortedLayers[0]->OutputDimensions()[0] == 25);
  REQUIRE(dynamic_cast<LinearNoBias<>*>(sortedLayers[1]) != nullptr);
  REQUIRE(sortedLayers[1]->OutputDimensions().size() == 1);
  REQUIRE(sortedLayers[1]->OutputDimensions()[0] == 10);
  REQUIRE(dynamic_cast<LinearNoBias<>*>(sortedLayers[2]) != nullptr);
  REQUIRE(sortedLayers[2]->OutputDimensions().size() == 1);
  REQUIRE(sortedLayers[2]->OutputDimensions()[0] == 3);

  arma::mat input, outputRef;
  mlpack::Load("pytorch_linear_no_bias_inputs.csv", input, Fatal);
  mlpack::Load("pytorch_linear_no_bias_outputs.csv", outputRef, Fatal);

  arma::mat output;
  dag.Predict(input, output);

  REQUIRE(approx_equal(output, outputRef, "both", 0.001, 0.001));
}

TEST_CASE("test_pytorch_linear", "[matching]")
{
  // Load the ONNX graph.
  onnx::GraphProto graph = GetGraph("pytorch_linear.onnx");
  DAGNetwork<> dag = SubgraphConvert(graph);

  REQUIRE(dag.Network().size() == 3);

  vector<Layer<>*> sortedLayers = dag.SortedNetwork();

  REQUIRE(sortedLayers.size() == 3);
  REQUIRE(dynamic_cast<Linear<>*>(sortedLayers[0]) != nullptr);
  REQUIRE(sortedLayers[0]->OutputDimensions().size() == 1);
  REQUIRE(sortedLayers[0]->OutputDimensions()[0] == 25);
  REQUIRE(dynamic_cast<Linear<>*>(sortedLayers[1]) != nullptr);
  REQUIRE(sortedLayers[1]->OutputDimensions().size() == 1);
  REQUIRE(sortedLayers[1]->OutputDimensions()[0] == 10);
  REQUIRE(dynamic_cast<Linear<>*>(sortedLayers[2]) != nullptr);
  REQUIRE(sortedLayers[2]->OutputDimensions().size() == 1);
  REQUIRE(sortedLayers[2]->OutputDimensions()[0] == 3);

  arma::mat input, outputRef;
  mlpack::Load("pytorch_linear_inputs.csv", input, Fatal);
  mlpack::Load("pytorch_linear_outputs.csv", outputRef, Fatal);

  arma::mat output;
  dag.Predict(input, output);

  REQUIRE(approx_equal(output, outputRef, "both", 0.001, 0.001));
}

TEST_CASE("test_pytorch_linear_dynamo", "[matching]")
{
  // Load the ONNX graph.
  onnx::GraphProto graph = GetGraph("pytorch_linear_dynamo.onnx");
  DAGNetwork<> dag = SubgraphConvert(graph);

  REQUIRE(dag.Network().size() == 3);

  vector<Layer<>*> sortedLayers = dag.SortedNetwork();

  REQUIRE(sortedLayers.size() == 3);
  REQUIRE(dynamic_cast<Linear<>*>(sortedLayers[0]) != nullptr);
  REQUIRE(sortedLayers[0]->OutputDimensions().size() == 1);
  REQUIRE(sortedLayers[0]->OutputDimensions()[0] == 25);
  REQUIRE(dynamic_cast<Linear<>*>(sortedLayers[1]) != nullptr);
  REQUIRE(sortedLayers[1]->OutputDimensions().size() == 1);
  REQUIRE(sortedLayers[1]->OutputDimensions()[0] == 10);
  REQUIRE(dynamic_cast<Linear<>*>(sortedLayers[2]) != nullptr);
  REQUIRE(sortedLayers[2]->OutputDimensions().size() == 1);
  REQUIRE(sortedLayers[2]->OutputDimensions()[0] == 3);

  arma::mat input, outputRef;
  mlpack::Load("pytorch_linear_inputs.csv", input, Fatal);
  mlpack::Load("pytorch_linear_outputs.csv", outputRef, Fatal);

  arma::mat output;
  dag.Predict(input, output);

  REQUIRE(approx_equal(output, outputRef, "both", 0.001, 0.001));
}

TEST_CASE("test_tf_linear_no_bias", "[matching]")
{
  // Load the ONNX graph.
  onnx::GraphProto graph = GetGraph("tf_linear_no_bias.onnx");
  DAGNetwork<> dag = SubgraphConvert(graph);

  REQUIRE(dag.Network().size() == 3);

  vector<Layer<>*> sortedLayers = dag.SortedNetwork();

  REQUIRE(sortedLayers.size() == 3);
  REQUIRE(dynamic_cast<LinearNoBias<>*>(sortedLayers[0]) != nullptr);
  REQUIRE(sortedLayers[0]->OutputDimensions().size() == 1);
  REQUIRE(sortedLayers[0]->OutputDimensions()[0] == 25);
  REQUIRE(dynamic_cast<LinearNoBias<>*>(sortedLayers[1]) != nullptr);
  REQUIRE(sortedLayers[1]->OutputDimensions().size() == 1);
  REQUIRE(sortedLayers[1]->OutputDimensions()[0] == 10);
  REQUIRE(dynamic_cast<LinearNoBias<>*>(sortedLayers[2]) != nullptr);
  REQUIRE(sortedLayers[2]->OutputDimensions().size() == 1);
  REQUIRE(sortedLayers[2]->OutputDimensions()[0] == 3);

  arma::mat input, outputRef;
  mlpack::Load("tf_linear_no_bias_inputs.csv", input, Fatal);
  mlpack::Load("tf_linear_no_bias_outputs.csv", outputRef, Fatal);

  arma::mat output;
  dag.Predict(input, output);

  REQUIRE(approx_equal(output, outputRef, "both", 0.001, 0.001));
}

TEST_CASE("test_tf_linear", "[matching]")
{
  // Load the ONNX graph.
  onnx::GraphProto graph = GetGraph("tf_linear.onnx");
  DAGNetwork<> dag = SubgraphConvert(graph);

  REQUIRE(dag.Network().size() == 3);

  vector<Layer<>*> sortedLayers = dag.SortedNetwork();

  REQUIRE(sortedLayers.size() == 3);
  REQUIRE(dynamic_cast<Linear<>*>(sortedLayers[0]) != nullptr);
  REQUIRE(sortedLayers[0]->OutputDimensions().size() == 1);
  REQUIRE(sortedLayers[0]->OutputDimensions()[0] == 25);
  REQUIRE(dynamic_cast<Linear<>*>(sortedLayers[1]) != nullptr);
  REQUIRE(sortedLayers[1]->OutputDimensions().size() == 1);
  REQUIRE(sortedLayers[1]->OutputDimensions()[0] == 10);
  REQUIRE(dynamic_cast<Linear<>*>(sortedLayers[2]) != nullptr);
  REQUIRE(sortedLayers[2]->OutputDimensions().size() == 1);
  REQUIRE(sortedLayers[2]->OutputDimensions()[0] == 3);

  arma::mat input, outputRef;
  mlpack::Load("tf_linear_inputs.csv", input, Fatal);
  mlpack::Load("tf_linear_outputs.csv", outputRef, Fatal);

  arma::mat output;
  dag.Predict(input, output);

  REQUIRE(approx_equal(output, outputRef, "both", 0.001, 0.001));
}

// Check that a network full of different activations can be converted.
TEST_CASE("test_pytorch_activations", "[matching]")
{
  onnx::GraphProto graph = GetGraph("pytorch_activations.onnx");
  DAGNetwork<> dag = SubgraphConvert(graph);

  REQUIRE(dag.Network().size() == 19);

  arma::mat input, outputRef;
  mlpack::Load("pytorch_activations_inputs.csv", input, Fatal);
  mlpack::Load("pytorch_activations_outputs.csv", outputRef, Fatal);

  arma::mat output;
  dag.Predict(input, output);

  output.print("output");
  outputRef.print("output_ref");

  REQUIRE(approx_equal(output, outputRef, "both", 0.001, 0.001));
}

TEST_CASE("test_pytorch_dynamo_activations", "[matching]")
{
  onnx::GraphProto graph = GetGraph("pytorch_activations_dynamo.onnx");
  DAGNetwork<> dag = SubgraphConvert(graph);

  REQUIRE(dag.Network().size() == 19);

  arma::mat input, outputRef;
  mlpack::Load("pytorch_activations_inputs.csv", input, Fatal);
  mlpack::Load("pytorch_activations_outputs.csv", outputRef, Fatal);

  arma::mat output;
  dag.Predict(input, output);

  REQUIRE(approx_equal(output, outputRef, "both", 0.001, 0.001));
}

TEST_CASE("test_tensorflow_activations", "[matching]")
{
  onnx::GraphProto graph = GetGraph("tf_activations.onnx");
  DAGNetwork<> dag = SubgraphConvert(graph);

  REQUIRE(dag.Network().size() == 21);

  arma::mat input, outputRef;
  mlpack::Load("tf_activations_inputs.csv", input, Fatal);
  mlpack::Load("tf_activations_outputs.csv", outputRef, Fatal);

  arma::mat output;
  dag.Predict(input, output);

  output.print("output");
  outputRef.print("outputRef");

  REQUIRE(approx_equal(output, outputRef, "both", 0.001, 0.001));
}

TEST_CASE("test_tensorflow_gelu_activation", "[matching]")
{
  onnx::GraphProto graph = GetGraph("tf_gelu_activation.onnx");
  DAGNetwork<> dag = SubgraphConvert(graph);

  REQUIRE(dag.Network().size() == 7);

  vector<Layer<>*> sortedLayers = dag.SortedNetwork();

  REQUIRE(sortedLayers.size() == 7);
  REQUIRE(dynamic_cast<Linear<>*>(sortedLayers[0]) != nullptr);
  REQUIRE(sortedLayers[0]->OutputDimensions().size() == 1);
  REQUIRE(sortedLayers[0]->OutputDimensions()[0] == 25);
  REQUIRE(dynamic_cast<GELUExact<>*>(sortedLayers[1]) != nullptr);
  REQUIRE(dynamic_cast<Linear<>*>(sortedLayers[2]) != nullptr);
  REQUIRE(sortedLayers[2]->OutputDimensions().size() == 1);
  REQUIRE(sortedLayers[2]->OutputDimensions()[0] == 25);
  REQUIRE(dynamic_cast<GELUExact<>*>(sortedLayers[3]) != nullptr);
  REQUIRE(dynamic_cast<Linear<>*>(sortedLayers[4]) != nullptr);
  REQUIRE(sortedLayers[4]->OutputDimensions().size() == 1);
  REQUIRE(sortedLayers[4]->OutputDimensions()[0] == 10);
  REQUIRE(dynamic_cast<GELUExact<>*>(sortedLayers[5]) != nullptr);
  REQUIRE(dynamic_cast<Linear<>*>(sortedLayers[6]) != nullptr);
  REQUIRE(sortedLayers[6]->OutputDimensions().size() == 1);
  REQUIRE(sortedLayers[6]->OutputDimensions()[0] == 3);


  arma::mat input, outputRef;
  mlpack::Load("tf_gelu_activation_inputs.csv", input, Fatal);
  mlpack::Load("tf_gelu_activation_outputs.csv", outputRef, Fatal);

  arma::mat output;
  dag.Predict(input, output);

  REQUIRE(approx_equal(output, outputRef, "both", 0.001, 0.001));
}
