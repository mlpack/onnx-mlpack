/**
 * @file lenet.cpp
 * @author Ryan Curtin
 *
 * Check that Lenet5 can be imported into mlpack.
 *
 * This particular version comes from
 * https://github.com/ONNC/onnc-tutorial/blob/master/models/lenet/lenet.onnx .
 *
 * NOTE: this graph actually has very little to do with Lenet5 itself!
 * The structure is:
 *
 *    Convolution
 *    ReLU
 *    Padding
 *    MaxPool
 *    Convolution
 *    ReLU
 *    Padding
 *    MaxPool
 *    Convolution
 *    ReLU
 *    Convolution
 *
 * which... doesn't have any dense layers at the end.
 *
 * In any case, it works for our purposes of testing, and we can check that we
 * get the same outputs after conversion.
 *
 * There are a couple other caveats of this graph that are also useful for
 * testing:
 *
 *  - The weights are stored as vectors, but then ONNX Reshape nodes are used
 *    before feeding them into layers.
 *  - There is a superfluous Identity ONNX node before the output.
 *  - The ONNX Conv nodes do not use the 'bias' input, but instead use a
 *    separate Add node to add in the bias.
 *
 * For the sake of testing our converter, weird is good.
 */
#include <onnx_mlpack.hpp>
#include "catch.hpp"

using namespace std;
using namespace mlpack;

TEST_CASE("lenet5_test", "[lenet]")
{
  // Get the mlpack model from the graph.
  DAGNetwork<> generatedModel = onnx_mlpack::Convert("onnc-lenet5.onnx");

  REQUIRE(generatedModel.Network().size() == 11);

  // Make sure we got the right structure.
  const std::vector<Layer<>*>& layers = generatedModel.SortedNetwork();
  REQUIRE(dynamic_cast<Convolution<>*>(layers[ 0]) != nullptr);
  REQUIRE(dynamic_cast<ReLU<>*       >(layers[ 1]) != nullptr);
  REQUIRE(dynamic_cast<Padding<>*    >(layers[ 2]) != nullptr);
  REQUIRE(dynamic_cast<MaxPooling<>* >(layers[ 3]) != nullptr);
  REQUIRE(dynamic_cast<Convolution<>*>(layers[ 4]) != nullptr);
  REQUIRE(dynamic_cast<ReLU<>*       >(layers[ 5]) != nullptr);
  REQUIRE(dynamic_cast<Padding<>*    >(layers[ 6]) != nullptr);
  REQUIRE(dynamic_cast<MaxPooling<>* >(layers[ 7]) != nullptr);
  REQUIRE(dynamic_cast<Convolution<>*>(layers[ 8]) != nullptr);
  REQUIRE(dynamic_cast<ReLU<>*       >(layers[ 9]) != nullptr);
  REQUIRE(dynamic_cast<Convolution<>*>(layers[10]) != nullptr);

  // Test running some inputs through the network.
  arma::mat inputData;
  Load("onnc_lenet5_inputs.csv", inputData, Fatal);

  arma::mat actualOutputs;
  generatedModel.Predict(inputData, actualOutputs);

  arma::mat outputData;
  Load("onnc_lenet5_outputs.csv", outputData, Fatal);

  REQUIRE(actualOutputs.n_cols == outputData.n_cols);
  REQUIRE(actualOutputs.n_rows == outputData.n_rows);
  REQUIRE(approx_equal(actualOutputs, outputData, "both", 0.001, 0.001));
}
