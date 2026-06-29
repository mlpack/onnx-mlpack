/**
 * @file convert_impl.hpp
 * @author Kumar Utkarsh
 * @author Ryan Curtin
 *
 * Implementation of ONNX-mlpack converter (top level).
 */
#ifndef ONNX_MLPACK_CONVERT_IMPL_HPP
#define ONNX_MLPACK_CONVERT_IMPL_HPP

#include "convert.hpp"
#include "apply_initial_reshapes.hpp"
#include "remove_useless_nodes.hpp"
#include "matchers/match.hpp"

#include <onnx/onnx_pb.h>
#include <onnx/onnx-ml.pb.h>
// It is deeply absurd that I have to do this.
#define ONNX_NAMESPACE onnx
#include <onnx/shape_inference/implementation.h>

namespace onnx_mlpack {

// Load an ONNX model from the specified path.
inline onnx::GraphProto GetGraph(const std::string &filePath)
{
  // ModelProto contains the ONNX graph along with some metadata.
  // We only need the graph from the ModelProto.
  onnx::ModelProto onnxModel;
  std::ifstream in(filePath, std::ios_base::binary);
  if (!in.is_open())
  {
    throw std::runtime_error("Failed to open ONNX model file '" + filePath +
        "'!");
  }
  // Parse the ONNX model from the input stream.
  onnxModel.ParseFromIstream(&in);
  in.close();

  // Perform shape inference on the model.
  onnx::shape_inference::InferShapes(onnxModel);

  // Apply any reshapes on inputs to the graph.
  onnx::GraphProto graph = onnxModel.graph();
  ApplyInitialReshapes(graph);

  // Remove any useless operators if possible.
  RemoveUselessNodes(graph);

  // Return the graph from the ONNX model.
  return graph;
}

// Convert the given ONNX graph into an mlpack DAGNetwork by iteratively
// matching subgraphs of the ONNX network.
inline mlpack::DAGNetwork<> Convert(const onnx::GraphProto& graph)
{
  // Before we start, we need to ensure that the graph has only one
  // input for which there isn't an initializer.
  std::set<std::string> initializerNames;
  for (size_t i = 0; i < graph.initializer_size(); ++i)
    if (graph.initializer(i).has_name())
      initializerNames.insert(graph.initializer(i).name());

  size_t inputsWithoutInitializers = 0;
  for (size_t i = 0; i < graph.input_size(); ++i)
  {
    if (graph.input(i).has_name() &&
        initializerNames.count(graph.input(i).name()) == 0)
      ++inputsWithoutInitializers;
    else if (!graph.input(i).has_name())
      ++inputsWithoutInitializers;
  }

  if (inputsWithoutInitializers != 1)
  {
    throw std::runtime_error("onnx_mlpack::Convert(): input ONNX graph must "
        "have one and only one input without an initializer!");
  }

  // First collect all of the subgraphs that we might be trying to match.
  std::vector<Subgraph*> subgraphs;
  subgraphs.push_back(new LinearNoBiasGemmSubgraph());
  subgraphs.push_back(new LinearNoBiasMatMulSubgraph());
  subgraphs.push_back(new LinearGemmSubgraph());
  subgraphs.push_back(new LinearMatMulAddSubgraph());
  subgraphs.push_back(new CELUSubgraph());
  subgraphs.push_back(new ELUSubgraph());
  subgraphs.push_back(new ELUPiecewiseSubgraph());
  subgraphs.push_back(new GELUExactSubgraph());
  subgraphs.push_back(new GELUExactMultiOpSubgraph());
  subgraphs.push_back(new GELUSubgraph());
  subgraphs.push_back(new GELUMultiOpSubgraph());
  subgraphs.push_back(new HardSigmoidSubgraph());
  subgraphs.push_back(new HardSigmoidMultiOpSubgraph());
  subgraphs.push_back(new HardSwishSubgraph());
  subgraphs.push_back(new LeakyReLUSubgraph());
  subgraphs.push_back(new MishSubgraph());
  subgraphs.push_back(new MishMultiOpSubgraph());
  subgraphs.push_back(new PReLUSubgraph());
  subgraphs.push_back(new PReLUMultiOpSubgraph());
  subgraphs.push_back(new ReLUSubgraph());
  subgraphs.push_back(new SELUSubgraph());
  subgraphs.push_back(new SigmoidSubgraph());
  subgraphs.push_back(new SoftplusSubgraph());
  subgraphs.push_back(new SoftplusThresholdSubgraph());
  subgraphs.push_back(new SwishSubgraph());
  subgraphs.push_back(new TanhSubgraph());
  subgraphs.push_back(new MaxPoolingSubgraph());
  subgraphs.push_back(new ConvSubgraph());
  subgraphs.push_back(new ConvAddSubgraph());
  subgraphs.push_back(new MulScalarSubgraph());
  subgraphs.push_back(new BatchNormSubgraph());
  subgraphs.push_back(new SoftmaxSubgraph());

  // Find the best subgraph match.
  const Matching m = Matcher(graph, subgraphs);

  // Next, for each subgraph match, we will convert the relevant layers to the
  // mlpack layer equivalent.
  //
  // TODO: handle template parameters for loss function?
  mlpack::DAGNetwork<> result;

  // Extract all the vertices of the mlpack network.  We have to track how many
  // mlpack layers were added during the conversion.
  arma::uvec layerCounts(m.matches.size());
  for (size_t i = 0; i < m.matches.size(); ++i)
  {
    const size_t origNetworkSize = result.Network().size();
    m.matches[i].second->Convert(m.matches[i].first, graph, result);
    const size_t newNetworkSize = result.Network().size();

    layerCounts[i] = (newNetworkSize - origNetworkSize);
  }

  // The convention is that the first layer that a subgraph match adds is the
  // input, and the last layer is the output.
  arma::uvec layerInputs(m.matches.size());
  arma::uvec layerOutputs(m.matches.size());
  layerOutputs[0] = layerCounts[0] - 1;
  for (size_t i = 1; i < m.matches.size(); ++i)
  {
    layerInputs[i] = layerOutputs[i - 1] + 1;
    layerOutputs[i] = layerOutputs[i - 1] + layerCounts[i];
  }

  // Now make connections between each of the vertices.
  std::vector<std::pair<size_t, size_t>> connections =
      FindConnections(m, graph);
  for (const std::pair<size_t, size_t>& p : connections)
    result.Connect(layerOutputs[p.first], layerInputs[p.second]);

  // Extract the size of the input.
  const onnx::ValueInfoProto& input = graph.input(0);
  if (!input.has_type() || !input.type().has_tensor_type())
  {
    throw std::runtime_error("onnx_mlpack::Convert(): ONNX graph input must be "
        "of tensor type!");
  }

  // ONNX generally uses the first dimension as the batch size; we need to
  // ignore that!  Other dimensions need to have their orders reversed, since
  // mlpack is column-major and ONNX is row-major.
  std::vector<size_t> inputDims(
      input.type().tensor_type().shape().dim_size() - 1);

  const size_t numDims = input.type().tensor_type().shape().dim_size();
  for (size_t i = 1; i < numDims; ++i)
  {
    if (!input.type().tensor_type().shape().dim(i).has_dim_value())
    {
      std::ostringstream oss;
      oss << "onnx_mlpack::Convert(): ONNX graph input dimension " << i
          << " does not have specified size; cannot convert!";
      throw std::runtime_error(oss.str());
    }
    inputDims[numDims - i - 1] =
        input.type().tensor_type().shape().dim(i).dim_value();
  }

  result.InputDimensions() = std::move(inputDims);
  result.Reset(); // Propagate the dimensions through the network.

  // Transfer all of the weights of each layer.
  for (size_t i = 0; i < m.matches.size(); ++i)
  {
    // We have to pass all layers that the subgraph created.
    std::vector<mlpack::Layer<>*> createdLayers;
    for (size_t j = layerInputs[i]; j <= layerOutputs[i]; ++j)
      createdLayers.push_back(result.Network()[j]);

    m.matches[i].second->TransferWeights(m.matches[i].first, graph,
        createdLayers);
  }

  // Clean up.
  for (size_t s = 0; s < subgraphs.size(); ++s)
  {
    delete subgraphs[s];
    subgraphs[s] = nullptr;
  }

  return result;
}

} // namespace onnx_mlpack

#endif
