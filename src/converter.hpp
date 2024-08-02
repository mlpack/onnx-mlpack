#ifndef CONVERTER_HPP
#define CONVERTER_HPP

#include "operations/add_layer.hpp"
#include "model_parser/attribute.hpp"


onnx::GraphProto getGraph(string filePath);

mlpack::FFN<> converter(onnx::GraphProto graph);
void printParametersSize(vector<arma::Mat<double>> layerParameters);
arma::mat FlattenParameters(vector<arma::Mat<double>> layerParameters);


#include "converter_impl.hpp"
#endif