#include "converter.hpp"

onnx::GraphProto getGraph(string filePath)
{
    onnx::ModelProto onnxModel;
    std::ifstream in(filePath, std::ios_base::binary);
    onnxModel.ParseFromIstream(&in);
    in.close();
    return onnxModel.graph();
}

mlpack::FFN<> converter(onnx::GraphProto graph)
{
    mlpack::FFN<> ffn;
    // at first layers will be added in ffn and corresponding parameters will be stored
    // in layerParameters, and once the whole model is set, the parameters inside
    // layerParameters will be transfered to ffn
    vector<arma::Mat<double>> layerParameters;
    string modelInput = get::ModelInput(graph);
    vector<size_t> inputDimension = get::InputDimension(graph, modelInput);
    ffn.InputDimensions() = inputDimension;
    cout << "inputDimension " << inputDimension << endl;

    // Iterating through nodes in topological order
    vector<int> topoSortedNode = get::TopologicallySortedNodes(graph);

    for (int nodeIndex : topoSortedNode)
    {
        if (nodeIndex == 117)
        {
            cout << "this is the point" << endl;
            int i = 3;
        }
        onnx::NodeProto node = graph.node(nodeIndex);
        // extracting the attributes and adding the layer to the ffn
        map<string, double> onnxOperatorAttribute = OnnxOperatorAttribute(graph, node);
        AddLayer(ffn, graph, node, onnxOperatorAttribute, layerParameters);
    }
    // -----------------------------
    ffn.Reset();
    printParametersSize(layerParameters);

    // vectorising the layer prameters and putting all together
    // and then transferring the whole parameters to the model
    // at once
    // arma::mat flattenParameters = FlattenParameters(layerParameters);

    // ffn.Parameters() = flattenParameters;

    // mapping the parameters to the layers
    int i = 0;
    for (mlpack::Layer<> *layer : ffn.Network())
    {
        if (layerParameters[i].n_elem)
        {
            // making rounding
            // int decimal_places = 20;
            // Scaling factor
            // double scale = std::pow(10.0, decimal_places);

            // Scale, round, and then scale back
            // arma::mat A = arma::round(layerParameters[i] * scale) / scale;
            // arma::mat A = layerParameters[i];

            // std::cout << std::fixed << std::setprecision(20);
            // A.submat(0, 0, 10, 0).raw_print(std::cout);

            // int rows = layer->Parameters().n_rows;
            // int cols = layer->Parameters().n_cols;
            // int _rows = layerParameters[i].n_rows;
            // int _cols = layerParameters[i].n_cols;
            // cout<< "layer parameters: [" << rows << ", "<< cols << " ] and stored parameters: [" <<_rows << ", " << _cols << " ]"<<endl;
            layer->Parameters() = layerParameters[i];
            cout << "layerParameters " << i << endl;
            // A.submat(0, 0, 10, 0).print(" ");
        }
        // cout<<i<<endl;
        i++;
    }
    return ffn;
}

// get the whole size of the parameters
void printParametersSize(vector<arma::Mat<double>> layerParameters)
{
    int count = 0;
    for (auto element : layerParameters)
    {
        count += (element.n_rows * element.n_cols);
    }
    cout << count << "<--------" << endl;
}

arma::mat FlattenParameters(vector<arma::Mat<double>> layerParameters)
{
    arma::vec flattenParameters;
    for (auto layerParameter : layerParameters)
    {
        flattenParameters = arma::join_vert(flattenParameters, arma::vectorise(layerParameter));
    }
    return flattenParameters;
}