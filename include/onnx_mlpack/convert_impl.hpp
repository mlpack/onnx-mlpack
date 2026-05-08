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

  // Return the graph from the ONNX model.
  return onnxModel.graph();
}

/**
 * Convert an ONNX graph to an mlpack FFN model.
 *
 * We iterate through the nodes of the ONNX graph in topological order, adding
 * the corresponding layers to the `mlpack::FFN` model and storing the
 * parameters in `layerParameters`. After adding all the layers to the FFN, we
 * call `ffn.Reset()` to ensure all layers adjust their input/output dimensions
 * and check for compatibility.
 *
 * Finally, we iterate through the layers again, transferring the corresponding
 * weights to the FFN layers.
 */
inline mlpack::DAGNetwork<> Convert(onnx::GraphProto& graph)
{
  mlpack::DAGNetwork<> dag;

  // Get the name of the first node to set the input dimensions of the DAG.
  std::string modelInput = ModelInput(graph);
  dag.InputDimensions() = InputDimension(graph, modelInput);

  // Get the nodes in topologically sorted order.
  std::vector<std::vector<size_t>> adj = AdjacencyMatrix(graph);
  std::vector<size_t> topoSortedNode = TopologicallySortedNodes(graph, adj);
  std::map<size_t, std::vector<size_t>> onnxLayerIndex_mlpackLayerIndex;

  // Iterate through the topologically sorted nodes.
  size_t index = 1;
  for (size_t nodeIndex : topoSortedNode)
  {
    // Get the actual node from its index.
    const onnx::NodeProto &node = graph.node(nodeIndex);

    std::cout << index << " " << node.name() << " => " << node.op_type()
        << std::endl;
    index++;
    // Extract the attributes from the node.
    std::map<std::string, double> onnxOperatorAttribute =
        OnnxOperatorAttribute(graph, node);

    // Use the attributes to generate an mlpack layer and add that layer to the
    // FFN.  This step adds the layer to the FFN and stores the parameters in
    // `layerParameters`.
    onnxLayerIndex_mlpackLayerIndex[nodeIndex] =
        AddLayer(dag, graph, node, onnxOperatorAttribute);
  }

  // Make connection in the DAG.
  for (size_t currOnnxNode = 0; currOnnxNode < adj.size(); currOnnxNode++)
  {
    std::vector<size_t> currMlpackLayer =
        onnxLayerIndex_mlpackLayerIndex[currOnnxNode];

    for (auto nextOnnxNode : adj[currOnnxNode])
    {
      std::vector<size_t> nextMlpackLayer =
          onnxLayerIndex_mlpackLayerIndex[nextOnnxNode];

      dag.Connect(
          currMlpackLayer[currMlpackLayer.size() - 1],
          nextMlpackLayer[0]);
    }
  }

  // Reset the FFN to ensure all layers adjust their input/output dimensions.
  dag.Reset();

  /*
  Method 2: Flatten the layer parameters, put them all together,
  and then transfer the whole parameters to the model at once.
  */
  // arma::mat flattenParameters = FlattenParameters(layerParameters);
  // ffn.Parameters() = flattenParameters;

  /*
  Method 1: Transfer the parameters to each layer one by one.
  */
  for (auto nodeIndex : topoSortedNode)
  {
    const onnx::NodeProto &node = graph.node(nodeIndex);
    std::map<std::string, double> onnxOperatorAttribute =
        OnnxOperatorAttribute(graph, node);
    TransferWeights(dag, onnxLayerIndex_mlpackLayerIndex[nodeIndex], graph,
        node, onnxOperatorAttribute);
    std::cout << "Transferred parameters to layer " << node.op_type()
        << std::endl;
  }

  return dag;
}

// Eventually this will replace Convert() overall.
inline mlpack::DAGNetwork<> SubgraphConvert(const onnx::GraphProto& graph)
{
  // Before we start, we need to ensure that the graph has only one input.
  if (graph.input_size() != 1)
  {
    throw std::runtime_error("SubgraphConvert(): input ONNX graph must have "
        "one and only one input!");
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

  // Find the best subgraph match.
  const Matching m = Matcher(graph, subgraphs);

  // Next, for each subgraph match, we will convert the relevant layers to the
  // mlpack layer equivalent.
  //
  // TODO: handle template parameters for loss function?
  mlpack::DAGNetwork<> result;

  // Extract all the vertices of the mlpack network.
  for (size_t i = 0; i < m.matches.size(); ++i)
  {
    m.matches[i].second->Convert(m.matches[i].first, graph, result);
  }

  // Now make connections between each of the vertices.
  std::vector<std::pair<size_t, size_t>> connections =
      FindConnections(m, graph);
  for (const std::pair<size_t, size_t>& p : connections)
    result.Connect(p.first, p.second);

  // Extract the size of the input.
  const onnx::ValueInfoProto& input = graph.input(0);
  if (!input.has_type() || !input.type().has_tensor_type())
  {
    throw std::runtime_error("SubgraphConvert(): ONNX graph input must be of "
        "tensor type!");
  }

  // ONNX generally uses the first dimension as the batch size; we need to
  // ignore that!
  std::vector<size_t> inputDims(
      input.type().tensor_type().shape().dim_size() - 1);

  for (size_t i = 1; i < input.type().tensor_type().shape().dim_size(); ++i)
  {
    if (!input.type().tensor_type().shape().dim(i).has_dim_value())
    {
      std::ostringstream oss;
      oss << "SubgraphConvert(): ONNX graph input dimension " << i << " does "
          << "not have specified size; cannot convert!";
      throw std::runtime_error(oss.str());
    }
    inputDims[i - 1] = input.type().tensor_type().shape().dim(i).dim_value();
  }

  result.InputDimensions() = std::move(inputDims);
  result.Reset(); // Propagate the dimensions through the network.

  // Transfer all of the weights of each layer.
  for (size_t i = 0; i < m.matches.size(); ++i)
  {
    m.matches[i].second->TransferWeights(m.matches[i].first, graph,
        result.Network()[i]);
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
