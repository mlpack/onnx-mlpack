#include "converter.hpp"

inline onnx::GraphProto getGraph(const std::string &filePath)
{
    // ModelProto contains the ONNX graph along with some metadata.
    // We only need the graph from the ModelProto.
    onnx::ModelProto onnxModel;
    std::ifstream in(filePath, std::ios_base::binary);
    if (!in.is_open())
    {
        throw std::runtime_error("Failed to open ONNX model file: " + filePath);
    }
    // Parse the ONNX model from the input stream.
    onnxModel.ParseFromIstream(&in);
    in.close();

    // Return the graph from the ONNX model.
    return onnxModel.graph();
}

/**
 * @brief Convert an ONNX graph to an mlpack FFN model.
 *
 * We iterate through the nodes of the ONNX graph in topological order,
 * adding the corresponding layers to the `mlpack::FFN` model and storing the
 * parameters in `layerParameters`. After adding all the layers to the FFN,
 * we call `ffn.Reset()` to ensure all layers adjust their input/output dimensions
 * and check for compatibility.
 *
 * Finally, we iterate through the layers again, transferring the corresponding
 * weights to the FFN layers.
 */
inline mlpack::DAGNetwork<> converter(onnx::GraphProto &graph)
{
    mlpack::DAGNetwork<> dag;

    // Buffer to store parameters of each layer for later use.
    // std::vector<arma::Mat<double>> layerParameters;

    // Get the name of the first node to set the input dimensions of the DAG.
    std::string modelInput = get::ModelInput(graph);
    std::vector<size_t> inputDimension = get::InputDimension(graph, modelInput);
    dag.InputDimensions() = inputDimension;

    // Get the nodes in topologically sorted order.
    vector<vector<int>> adj = get::AdjacencyMatrix(graph);
    std::vector<int> topoSortedNode = get::TopologicallySortedNodes(graph, adj);
    // std::vector<int> mlpackLayerIndex(topoSortedNode.size());
    map<int, vector<size_t>> onnxLayerIndex_mlpackLayerIndex;

    // Iterate through the topologically sorted nodes.
    cout << endl
         << "****GENERATING THE GRAPH****" << endl;

    int index = 1;

    for (int nodeIndex : topoSortedNode)
    {
        // Get the actual node from its index.
        const onnx::NodeProto &node = graph.node(nodeIndex);

        cout<<index<<" "<<node.name()<<" => "<<node.op_type()<<endl;
        index++;
        // Extract the attributes from the node.
        std::map<std::string, double> onnxOperatorAttribute = OnnxOperatorAttribute(graph, node);

        // Use the attributes to generate an mlpack layer and add that layer to the FFN.
        // This step adds the layer to the FFN and stores the parameters in `layerParameters`.
        onnxLayerIndex_mlpackLayerIndex[nodeIndex] = AddLayer(dag, graph, node, onnxOperatorAttribute);
    }
    // exit(0);

    // make connection in the dag
    for (int currOnnxNode = 0; currOnnxNode < adj.size(); currOnnxNode++)
    {
        vector<size_t> currMlpackLayer = onnxLayerIndex_mlpackLayerIndex[currOnnxNode];

        for (auto nextOnnxNode : adj[currOnnxNode])
        {
            vector<size_t> nextMlpackLayer = onnxLayerIndex_mlpackLayerIndex[nextOnnxNode];

            dag.Connect(
                currMlpackLayer[currMlpackLayer.size() - 1],
                nextMlpackLayer[0]);
        }
    }
    // Reset the FFN to ensure all layers adjust their input/output dimensions.
    dag.Reset();

    // exit(0);
    // printParametersSize(layerParameters);

    /*
    Method 2: Flatten the layer parameters, put them all together,
    and then transfer the whole parameters to the model at once.
    */
    // arma::mat flattenParameters = FlattenParameters(layerParameters);
    // ffn.Parameters() = flattenParameters;

    /*
    Method 1: Transfer the parameters to each layer one by one.
    */
    // cout << endl;
    cout << "****TRANSFERRING PARAMETERS TO THE LAYER****" << endl;
    for (auto nodeIndex : topoSortedNode)
    {

        const onnx::NodeProto &node = graph.node(nodeIndex);
        std::map<std::string, double> onnxOperatorAttribute = OnnxOperatorAttribute(graph, node);
        TransferWeights(dag, onnxLayerIndex_mlpackLayerIndex[nodeIndex], graph, node, onnxOperatorAttribute);
        std::cout << "Transferred parameters to layer " << node.op_type() << std::endl;
    }
    cout << endl;

    return dag;
}

// void printParametersSize(const std::vector<int> &mlpackLayerIndex)
// {
//     size_t count = 0;
//     for (const auto &element : layerParameters)
//     {
//         count += element.n_rows * element.n_cols;
//     }
//     std::cout << "Total parameter size: " << count << std::endl;
// }

// arma::mat FlattenParameters(const std::vector<int> &mlpackLayerIndex)
// {
//     arma::vec flattenParameters;
//     for (const auto &layerParameter : layerParameters)
//     {
//         // Concatenate the vectorized layer parameters into a single vector.
//         flattenParameters = arma::join_vert(flattenParameters, arma::vectorise(layerParameter));
//     }
//     return flattenParameters;
// }
