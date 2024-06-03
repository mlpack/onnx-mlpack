#ifndef CONVERTER_HPP
#define CONVERTER_HPP

#include "operations/add_layer.hpp"
#include "model_parser/attribute.hpp"

onnx::GraphProto getGraph(string filePath);
mlpack::FFN<> converter(onnx::GraphProto graph);
// converter(onnx::GraphProto graph);


#include "converter_impl.hpp"
#endif