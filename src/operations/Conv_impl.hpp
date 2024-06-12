#include "Conv.hpp"

void AddConv(mlpack::FFN<> &ffn, onnx::GraphProto graph,
             onnx::NodeProto node, map<string, double> onnxOperatorAttribute)
{
    // converting the onnx attribute to mlpack layer parameters
    size_t maps = FindConvMap(graph, node);
    size_t kernelHeight = onnxOperatorAttribute["kernel_shape_height"];
    size_t kernelWidth = onnxOperatorAttribute["kernel_shape_width"];
    size_t strideHeight = onnxOperatorAttribute["stride_height"];
    size_t strideWidth = onnxOperatorAttribute["stride_width"];
    size_t padW = 0;
    size_t padH = 0;
    string paddingType = "none";
    bool useBias = false;
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
        size_t padH = (onnxOperatorAttribute["pad_right"] + onnxOperatorAttribute["pad_left"]) / 2;
        size_t padW = (onnxOperatorAttribute["pad_top"] + onnxOperatorAttribute["pad_bottom"]) / 2;
    }

    mlpack::Convolution* Convolution = new mlpack::Convolution(maps, kernelWidth, kernelHeight, strideWidth, strideHeight, padW, padH, paddingType, useBias);
    ffn.Add(new mlpack::Identity());
    cout << "Added the Conv layer" << endl;
}



int FindConvMap(onnx::GraphProto graph, onnx::NodeProto node){
    return 0;      
}