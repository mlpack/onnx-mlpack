#include "Reshape.hpp"

void AddReshape(mlpack::FFN<> &ffn, onnx::GraphProto graph,
                           onnx::NodeProto node, map<string, double> onnxOperatorAttribute, vector<arma::Mat<double>> &layerParameters)
{
    // mlpack::Convolution* convolution = new mlpack::Convolution(maps, kernelWidth, kernelHeight, strideWidth, strideHeight, padW, padH, paddingType, useBias);
    mlpack::Identity* identity = new mlpack::Identity();

    // convolution->Parameters() = get::ConvertToColumnMajor(initializer);
    layerParameters.push_back(arma::Mat<double>());
    ffn.Add(identity);
    cout << "Added the Reshape layer" << endl;
}

vector<size_t> FindReshapedDimension(onnx::GraphProto graph, onnx::NodeProto node){

    string initializerName = node.input(1); // from this initializer we will be getting the dimension of the output of layer
    onnx::TensorProto initializer = get::Initializer(graph, initializerName);

    vector<size_t> dimensions(4, 1); /// colMajor W, H, C, C

    int j = 0;
    for(int i = initializer.dims(0)-1; i>=0; i--){
        dimensions[j] = initializer.int64_data(i);
        j++;
    }

    return dimensions;
}