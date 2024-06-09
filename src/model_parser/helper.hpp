#ifndef helper_HPP
#define helper_HPP

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "onnx_pb.h"

using namespace std;

namespace get{
    string ModelInput(onnx::GraphProto graph);
    vector<size_t> InputDimension(onnx::GraphProto graph, string modelInput);
    onnx::NodeProto CurrentNode(onnx::GraphProto graph, string nodeInput);
    onnx::TensorProto Initializer(onnx::GraphProto graph, string initializerName);
}

#include "helper_impl.hpp"
#endif
