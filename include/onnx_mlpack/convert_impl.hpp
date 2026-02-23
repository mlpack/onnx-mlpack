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

} // namespace onnx_mlpack

#endif
