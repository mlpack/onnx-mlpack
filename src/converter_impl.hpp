#include "converter.hpp"

onnx::GraphProto getGraph(const std::string &filePath)
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
mlpack::FFN<> converter(onnx::GraphProto &graph)
{
    mlpack::FFN<> ffn;

    // Buffer to store parameters of each layer for later use.
    std::vector<arma::Mat<double>> layerParameters;

    // Get the name of the first node to set the input dimensions of the FFN.
    std::string modelInput = get::ModelInput(graph);
    std::vector<size_t> inputDimension = get::InputDimension(graph, modelInput);
    ffn.InputDimensions() = inputDimension;

    // Get the nodes in topologically sorted order.
    std::vector<int> topoSortedNode = get::TopologicallySortedNodes(graph);

    // Iterate through the topologically sorted nodes.
    cout << endl
         << "****GENERATING THE GRAPH****" << endl;
    for (int nodeIndex : topoSortedNode)
    {
        // Get the actual node from its index.
        onnx::NodeProto node = graph.node(nodeIndex);

        // Extract the attributes from the node.
        std::map<std::string, double> onnxOperatorAttribute = OnnxOperatorAttribute(graph, node);

        // Use the attributes to generate an mlpack layer and add that layer to the FFN.
        // This step adds the layer to the FFN and stores the parameters in `layerParameters`.
        AddLayer(ffn, graph, node, onnxOperatorAttribute, layerParameters);
    }
    // Reset the FFN to ensure all layers adjust their input/output dimensions.
    ffn.Reset();
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
    cout << endl;
    cout << "****TRANSFERRING PARAMETERS TO THE LAYER****" << endl;
    for (size_t i = 0; i < ffn.Network().size(); ++i)
    {
        if (layerParameters[i].n_elem)
        {
            ffn.Network()[i]->Parameters() = layerParameters[i];
            std::cout << "Transferred parameters to layer " << i << std::endl;
        }
    }
    cout << endl;

    return ffn;
}

void printParametersSize(const std::vector<arma::Mat<double>> &layerParameters)
{
    size_t count = 0;
    for (const auto &element : layerParameters)
    {
        count += element.n_rows * element.n_cols;
    }
    std::cout << "Total parameter size: " << count << std::endl;
}

arma::mat FlattenParameters(const std::vector<arma::Mat<double>> &layerParameters)
{
    arma::vec flattenParameters;
    for (const auto &layerParameter : layerParameters)
    {
        // Concatenate the vectorized layer parameters into a single vector.
        flattenParameters = arma::join_vert(flattenParameters, arma::vectorise(layerParameter));
    }
    return flattenParameters;
}
