/**
 * @file convert_impl.hpp
 * @author Kumar Utkarsh
 * @author Ryan Curtin
 *
 * Implementation of ONNX-mlpack converter (top level).
 *
 * The ONNX/mlpack converter is free software; you may redistribute it and/or
 * modify it under the terms of the 3-clause BSD license.  You should have
 * received a copy of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ONNX_MLPACK_CONVERT_IMPL_HPP
#define ONNX_MLPACK_CONVERT_IMPL_HPP

#include "convert.hpp"
#include "apply_initial_reshapes.hpp"
#include "remove_useless_nodes.hpp"
#include "matchers/match.hpp"
#include "log.hpp"

#include <onnx/onnx_pb.h>
#include <onnx/onnx-ml.pb.h>
// It is deeply absurd that I have to do this.
#define ONNX_NAMESPACE onnx
#include <onnx/shape_inference/implementation.h>

namespace onnx_mlpack {

// Load an ONNX network and apply shape inference.
inline onnx::GraphProto Load(const std::string& filename)
{
  // ModelProto contains the ONNX graph along with some metadata.
  // We only need the graph from the ModelProto.
  onnx::ModelProto onnxModel;
  std::ifstream in(filename, std::ios_base::binary);
  if (!in.is_open())
  {
    throw std::runtime_error("onnx_mlpack::Load(): Failed to open ONNX model "
        "file '" + filename + "'!");
  }
  // Parse the ONNX model from the input stream.
  onnxModel.ParseFromIstream(&in);
  in.close();

  // Perform shape inference on the model.
  onnx::shape_inference::InferShapes(onnxModel);

  return onnxModel.graph();
}

// Simplify the structure of an ONNX model.
inline void Simplify(onnx::GraphProto& graph)
{
  // Apply any reshapes on inputs to the graph.
  ApplyInitialReshapes(graph);

  // Remove any useless operators if possible.
  RemoveUselessNodes(graph);
}

// Load and preprocess the given ONNX graph, and convert into an mlpack
// DAGNetwork.
inline mlpack::DAGNetwork<> Convert(const std::string& filename,
                                    const size_t logLevel)
{
  onnx::GraphProto graph = Load(filename);
  Simplify(graph);
  return Convert(graph, logLevel);
}

// Convert the given ONNX graph into an mlpack DAGNetwork by iteratively
// matching subgraphs of the ONNX network.
inline mlpack::DAGNetwork<> Convert(const onnx::GraphProto& graph,
                                    const size_t logLevel)
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
  subgraphs.push_back(new MeanPoolingSubgraph());

  // NOTE: this rule should be last!  Otherwise, internal Add connections inside
  // of a more complex layer may not be recognized correctly.
  subgraphs.push_back(new AddConnectionSubgraph());

  // Find the best subgraph match.
  const Matching m = Matcher(graph, subgraphs, logLevel);

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

  // Find all the connections between subgraphs.
  std::vector<std::pair<size_t, size_t>> connections =
      FindConnections(m, graph);

  // Collect which of the matches are the special AddConnection subgraphs.
  // The first element in the pair is the index in m.matches of the
  // AddConnection subgraph, and the second is the index of the subgraph in
  // m.matches that it is connected to.
  std::unordered_map<size_t, std::vector<size_t>> addConnections;
  for (size_t i = 0; i < m.matches.size(); ++i)
  {
    const std::string name = m.matches[i].second->Name();
    if (name == "AddConnection")
    {
      addConnections[i] = {};

      // Find what the AddConnection is connected to.
      for (const std::pair<size_t, size_t>& p : connections)
        if (p.first == i)
          addConnections[i].push_back(p.second);

      if (addConnections[i].size() == 0)
      {
        std::ostringstream oss;
        oss << "SubgraphConvert(): AddConnection subgraph " << i << " has no "
            << "output connection!";
        throw std::runtime_error(oss.str());
      }
    }
  }

  // Now make connections between each of the vertices.
  for (const std::pair<size_t, size_t>& p : connections)
    if (m.matches[p.first].second->Name() == "AddConnection")
      addConnections[p.first].push_back(p.second);

  // Each pair `p` connects the output of the subgraph index `p.first` to the
  // input of the subgraph index `p.second`.  However, there is one tricky
  // case: when the subgraph is an `AddConnection`, then no layer was actually
  // added to the DAGNetwork.  Instead, we need to connect the inputs of the
  // `AddConnection` to whatever the *next* layer is, and mark the connection
  // as an "ADDITION" connection.
  for (std::pair<const size_t, std::vector<size_t>>& p : addConnections)
  {
    // If any outputs of the AddConnection p.first are connected to another
    // AddConnection, we have to instead add the outputs of the target
    // AddConnection, and repeat this until there are no more AddConnections
    // in the list.
    bool updated = true;
    while (updated)
    {
      std::unordered_set<size_t> addConnectionIndices;
      std::unordered_set<size_t> newConnections;

      for (const size_t& t : p.second)
      {
        if (addConnections.count(t) > 0)
          addConnectionIndices.insert(t);
        else
          newConnections.insert(t);
      }

      if (addConnectionIndices.size() == 0)
        updated = false;

      for (const size_t& t : addConnectionIndices)
        for (const size_t& tt : addConnections[t])
          newConnections.insert(tt);

      p.second.clear();
      for (const size_t& t : newConnections)
        p.second.push_back(t);
    }
  }

  // Any layers with an AddConnection as input should have the ADDITION
  // connection type.
  for (const std::pair<const size_t, std::vector<size_t>>& p : addConnections)
    for (const size_t& target : p.second)
      result.SetConnection(target, mlpack::ConnectionTypes::ADDITION);

  // Now, replace any connections between an AddConnection and another layer
  // with whatever the actual connection should be.  (There could be multiple
  // connections that we have to add for a single AddConnection.)
  std::vector<std::pair<size_t, size_t>> finalConnections;
  for (const std::pair<size_t, size_t>& p : connections)
  {
    if (addConnections.count(p.first) > 0)
    {
      continue;
    }
    else if (addConnections.count(p.second) > 0)
    {
      for (const size_t& t : addConnections[p.second])
        finalConnections.push_back(std::make_pair(p.first, t));
    }
    else
    {
      finalConnections.push_back(p);
    }
  }

  for (const std::pair<size_t, size_t>& p : finalConnections)
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
