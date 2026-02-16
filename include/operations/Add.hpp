#ifndef ADD_HPP
#define ADD_HPP

#include "mlpack.hpp"
#include "onnx_pb.h"

using namespace std;

inline vector<size_t> AddAdd(mlpack::DAGNetwork<> &dag, onnx::GraphProto &graph,
            const onnx::NodeProto &node, map<string, double> onnxOperatorAttribute);


#include "Add_impl.hpp"
#endif
