#ifndef ATTRIBUTE_HPP
#define ATTRIBUTE_HPP

#include <iostream>
#include "helper.hpp"
#include "utils.hpp"
#include "onnx_pb.h"

using namespace std;


map<string, double> OnnxOperatorAttribute(onnx::GraphProto graph, onnx::NodeProto node);

#include "attribute_impl.hpp"
#endif