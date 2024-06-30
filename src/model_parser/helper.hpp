#ifndef helper_HPP
#define helper_HPP

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "queue"
#include "onnx_pb.h"
#include "stack"

using namespace std;

namespace get{
    string ModelInput(onnx::GraphProto graph);
    vector<size_t> InputDimension(onnx::GraphProto graph, string modelInput);
    // more than one node can take same input in a graph
    vector<int> CurrentNode(onnx::GraphProto graph, string nodeInput);
    onnx::TensorProto Initializer(onnx::GraphProto graph, string initializerName);
    arma::mat ConvertToColumnMajor(onnx::TensorProto initializer);

    // making the topological short with 
    vector<int> TopologicallySortedNodes(onnx::GraphProto &graph);
}
void dfs(int node, int visitedNode[], vector<vector<int>> adj, stack<int> &st);

#include "helper_impl.hpp"
#endif
