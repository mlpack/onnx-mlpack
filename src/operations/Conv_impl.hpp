#include "Conv.hpp"

void AddConv(mlpack::FFN<> &ffn, onnx::GraphProto graph,
              onnx::NodeProto node, map<string, double> onnxOperatorAttribute, vector<arma::Mat<double>> &layerParameters)
{
    bool useBias = false;
    onnx::TensorProto initializerWeights;
    onnx::TensorProto initializerBias;

    if(node.input().size() == 2){
        string initializerWeightsName = node.input(1);
        useBias = false;
        initializerWeights = get::Initializer(graph, initializerWeightsName);
        layerParameters.push_back(arma::conv_to<arma::mat>::from(get::ConvertToColumnMajor(initializerWeights)));
    }
    else if(node.input().size() == 3){
        string initializerWeightsName = node.input(1);
        string initializerBiasName = node.input(2);
        useBias = true;
        initializerWeights = get::Initializer(graph, initializerWeightsName);
        initializerBias = get::Initializer(graph, initializerBiasName);
        layerParameters.push_back(arma::join_cols(
            arma::conv_to<arma::mat>::from(get::ConvertToColumnMajor(initializerWeights)), 
            arma::conv_to<arma::mat>::from(get::ConvertToColumnMajor(initializerBias))
            ));
    }

    // converting the onnx attribute to mlpack layer parameters
    // size_t maps = FindConvMap(ffn, graph, node);
    size_t maps = initializerWeights.dims(0);
    size_t kernelHeight = onnxOperatorAttribute["kernel_height"];
    size_t kernelWidth = onnxOperatorAttribute["kernel_width"];
    size_t strideHeight = onnxOperatorAttribute["stride_height"];
    size_t strideWidth = onnxOperatorAttribute["stride_width"];
    size_t group = onnxOperatorAttribute["group"];
    size_t padW = 0;
    size_t padH = 0;
    string paddingType = "none";
    if(onnxOperatorAttribute["auto_pad_or_pads"] == 0) // auto_pad
    {
        if(onnxOperatorAttribute["auto_pad"] == 0) // NOT_SET => explicit value will be used
        {
            paddingType = "none";
        }
        if(onnxOperatorAttribute["auto_pad"] == 1 || onnxOperatorAttribute["auto_pad"] == 1) // SAME_UPPER OR SAME_LOWER
        {
            paddingType = "same";
        }
        if(onnxOperatorAttribute["auto_pad"] == 3) // VALID
        {
            paddingType = "valid";
        }
    }
    else if(onnxOperatorAttribute["auto_pad_or_pads"] == 1) // pads
    {
        padW = (onnxOperatorAttribute["pad_right"] + onnxOperatorAttribute["pad_left"]) / 2;
        padH = (onnxOperatorAttribute["pad_top"] + onnxOperatorAttribute["pad_bottom"]) / 2;
    }

    // layerParameters.push_back(arma::conv_to<arma::mat>::from(get::ConvertToColumnMajor(initializer)));
    
    if(group == 1){
        mlpack::Convolution* convolution = new mlpack::Convolution(maps, kernelWidth, kernelHeight, strideWidth, strideHeight, padW, padH, paddingType, useBias);
        ffn.Add(convolution);
        cout << "Added mlpack::Conv Layer" << endl;
    }else{
        mlpack::GroupedConvolution* convolution = new mlpack::GroupedConvolution(maps, kernelWidth, kernelHeight, group, strideWidth, strideHeight, padW, padH, paddingType, useBias);
        ffn.Add(convolution);
        cout << "Added mlpack::GroupedConv Layer" << endl;
    }
}

int FindConvMap(mlpack::FFN<> &ffn, onnx::GraphProto graph, onnx::NodeProto node){
    // return ffn.Network().back()->OutputDimensions()[2];     
    return 16;
}