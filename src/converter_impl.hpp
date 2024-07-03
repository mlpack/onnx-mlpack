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
    // layerParameters will be set once the whone network is created
    vector<arma::Mat<double>> layerParameters;
    string modelInput = get::ModelInput(graph);
    vector<size_t> inputDimension = get::InputDimension(graph, modelInput);
    ffn.InputDimensions() = inputDimension;
    cout<<"inputDimension "<< inputDimension<<endl;

    // Iterating through nodes in topological order
    vector<int> topoSortedNode = get::TopologicallySortedNodes(graph);

    for(int nodeIndex : topoSortedNode){
        if(nodeIndex == 117){
            cout<< "this is the point"<<endl;
            int i = 3;
        }
        onnx::NodeProto node = graph.node(nodeIndex);
        // extracting the attributes and adding the layer to the ffn
        map<string, double> onnxOperatorAttribute = OnnxOperatorAttribute(graph, node);
        AddLayer(ffn, graph, node, onnxOperatorAttribute, layerParameters);
    }
    ffn.Reset();
    // mapping the parameters to the layers
    int i = 0;
    for(mlpack::Layer<>* layer : ffn.Network()){
        if(layerParameters[i].n_elem){
            // int rows = layer->Parameters().n_rows;
            // int cols = layer->Parameters().n_cols;
            // int _rows = layerParameters[i].n_rows;
            // int _cols = layerParameters[i].n_cols;
            // cout<< "layer parameters: [" << rows << ", "<< cols << " ] and stored parameters: [" <<_rows << ", " << _cols << " ]"<<endl;
            layer->Parameters() = layerParameters[i];
        }
        // cout<<i<<endl;
        i++;
    }
    return ffn;
}