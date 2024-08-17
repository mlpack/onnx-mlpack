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
    string ModelInput(onnx::GraphProto &graph);
    vector<size_t> InputDimension(onnx::GraphProto &graph, string modelInput);
    // more than one node can take same input in a graph
    vector<int> CurrentNode(onnx::GraphProto &graph, string nodeInput);
    const onnx::TensorProto &Initializer(onnx::GraphProto &graph, string initializerName);
    arma::fmat ConvertToColumnMajor(const onnx::TensorProto &initializer);

    // making the topological short with 
    vector<vector<int>> AdjencyMatrix(onnx::GraphProto &graph);
    vector<int> TopologicallySortedNodes(onnx::GraphProto &graph);
    vector<double> convertToRowMajor(arma::mat matrix, vector<size_t> outputDimension);
    vector<double> convertToColMajor(arma::mat matrix, vector<size_t> outputDimension);
    void DrawRectangle(string imagePath, string finalImagePath, int r1, int c1, int r2, int c2, vector<int> imageDimension);
    void DrawRectangle_onCsv(arma::mat &matrix, int r1, int c1, int r2, int c2, vector<int> imageDimension);
}
void dfs(int node, int visitedNode[], vector<vector<int>> adj, stack<int> &st);

#include "helper_impl.hpp"
#endif
