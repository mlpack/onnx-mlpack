#include "MaxPool.hpp"

void AddMaxPool(mlpack::FFN<> &ffn, onnx::GraphProto graph,
                onnx::NodeProto node, map<string, double> onnxOperatorAttribute, vector<arma::Mat<double>> &layerParameters)
{
    size_t kernelWidth = onnxOperatorAttribute["kernel_width"];
    size_t kernelHeight = onnxOperatorAttribute["kernel_height"];
    size_t strideWidth = onnxOperatorAttribute["stride_width"];
    size_t strideHeight = onnxOperatorAttribute["stride_height"];
    bool floor = true;
    if (onnxOperatorAttribute["ceil_mode"] == 1)
    {
        floor = false;
    }

    // just to get the dimension of the incoming i will be reseting the ffn
    mlpack::FFN<> ffn_ = ffn;
    ffn_.Reset();
    vector<size_t>  dims = ffn_.Network().back()->OutputDimensions();
    // padding before maxPooling
    // size_t totalVerticalPadding = (strideWidth - 1) * dims[0] + kernelWidth - strideWidth;
    // size_t totalHorizontalPadding = (strideHeight - 1) * dims[1] + kernelHeight - strideHeight;
    size_t totalVerticalPadding = kernelWidth - strideWidth;
    size_t totalHorizontalPadding = kernelHeight - strideHeight;

    size_t padWLeft = totalVerticalPadding / 2;
    size_t padWRight = totalVerticalPadding - totalVerticalPadding / 2;
    size_t padHTop = totalHorizontalPadding / 2;
    size_t padHBottom = totalHorizontalPadding - totalHorizontalPadding / 2;

    mlpack::Padding *padding = new mlpack::Padding(padWLeft, padWRight, padHTop, padHBottom);
    // mlpack::Padding *padding = new mlpack::Padding(0, 0, 0, 0);
    layerParameters.push_back(arma::Mat<double>());
    ffn.Add(padding);
    cout << "Added mlpack::Padding Layer" << endl;


    // max pooling part
    mlpack::MaxPooling *maxPooling = new mlpack::MaxPooling(kernelWidth, kernelHeight, strideWidth, strideHeight, floor);
    layerParameters.push_back(arma::Mat<double>());
    ffn.Add(maxPooling);
    cout << "Added mlpack::MaxPool Layer" << endl;
    // vector<size_t> vec = {kernelWidth, kernelHeight, strideWidth, strideHeight};
    // cout << vec << endl;
}