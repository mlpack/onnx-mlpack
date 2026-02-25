/**
 * @file helper_impl.hpp
 * @author Kumar Utkarsh
 *
 * Implementation of helper utility functions.
 */
#ifndef ONNX_MLPACK_HELPER_IMPL_HPP
#define ONNX_MLPACK_HELPER_IMPL_HPP

#include "helper.hpp"

namespace onnx_mlpack {

inline std::string ModelInput(onnx::GraphProto& graph)
{
  std::vector<std::string> inputNames;
  std::vector<std::string> initializerNames;
  for (const onnx::ValueInfoProto& input : graph.input())
    inputNames.push_back(input.name());

  for (const onnx::TensorProto& initializer : graph.initializer())
    initializerNames.push_back(initializer.name());

  // input for which no initializer will be the modelInput
  for (const std::string& element : inputNames)
  {
    if (std::find(initializerNames.begin(), initializerNames.end(),
                  element) == initializerNames.end())
    {
      return element;
    }
  }

  // **** have to put error condition here
  std::cout << "error in finding the modelInput" << std::endl;
  return "";
}

inline std::vector<size_t> InputDimension(onnx::GraphProto &graph,
                                          const std::string &modelInput)
{
  std::vector<size_t> dimension;

  for (const onnx::ValueInfoProto& input : graph.input())
  {
    if (input.name() == modelInput)
    {
      const size_t dimSize = input.type().tensor_type().shape().dim().size();
      for (size_t i = dimSize - 1; i > 0; i--)
      {
        dimension.push_back(
            input.type().tensor_type().shape().dim(i).dim_value());
      }
    }
  }

  return dimension;
}

// to get the indexs of all the nodes which will be taking the output of the
// current node
inline std::vector<size_t> DependentNodes(onnx::GraphProto& graph,
                                          const std::string& nodeInput)
{
  std::vector<size_t> nodes;

  for (int i = 0; i < graph.node().size(); i++)
    for (int j = 0; j < graph.node(i).input().size(); j++)
      if (nodeInput == graph.node(i).input(j))
        nodes.push_back(i);

  return nodes;
}

inline std::vector<std::vector<size_t>> AdjacencyMatrix(onnx::GraphProto& graph)
{
  int totalNodes = graph.node().size();

  // creating the adjacency list and inDegree
  std::vector<std::vector<size_t>> adj(totalNodes);

  for (int i = 0; i < totalNodes; i++)
  {
    for (std::string output : graph.node(i).output())
      adj[i] = DependentNodes(graph, output);
  }

  return adj;
}

inline std::vector<size_t> TopologicallySortedNodes(
    onnx::GraphProto& graph,
    std::vector<std::vector<size_t>>& adj)
{
  const int totalNodes = graph.node().size();

  std::vector<size_t> visitedNode(totalNodes, 0);
  std::stack<size_t> st;

  for (int i = 0; i < totalNodes; i++)
  {
    if (!visitedNode[i])
      dfs(i, visitedNode, adj, st);
  }

  std::vector<size_t> topologicalSort;
  while (!st.empty())
  {
    topologicalSort.push_back(st.top());
    st.pop();
  }

  return topologicalSort;
}

inline void dfs(size_t node,
                std::vector<size_t>& visited,
                const std::vector<std::vector<size_t>>& adj,
                std::stack<size_t>& st)
{
  visited[node] = 1;
  for (size_t neighbouringNode : adj[node])
  {
    if (!visited[neighbouringNode])
      dfs(neighbouringNode, visited, adj, st);
  }

  st.push(node);
}

inline const onnx::TensorProto& Initializer(onnx::GraphProto& graph,
                                            const std::string& initializerName)
{
  for (const onnx::TensorProto& init : graph.initializer())
  {
    if (initializerName == init.name())
      return init;
  }

  throw std::runtime_error("No initializer found with name " + initializerName);
}

inline arma::fmat ConvertToColumnMajor(const onnx::TensorProto& initializer)
{
  // onnx initializer stores data in row major format
  // {N, C, H, W}
  std::vector<int> rowMajorDim(4, 1);
  int j = 3;
  const int nDims = initializer.dims().size();
  for (int i = nDims - 1; i >= 0; i--)
  {
    rowMajorDim[j] = initializer.dims(i);
    j--;
  }

  const int N = rowMajorDim[0]; // l
  const int C = rowMajorDim[1]; // k
  const int H = rowMajorDim[2]; // j
  const int W = rowMajorDim[3]; // i
  std::vector<float> colMajorData;

  for (size_t l = 0; l < N; l++)
  {
    for (size_t k = 0; k < C; k++)
    {
      for (size_t j = 0; j < W; j++)
      {
        for (size_t i = 0; i < H; i++)
        {
          // size_t colMajorIndex = l * (H * C * N) + k * (C * N) + j * (N) + i;
          int rowMajorIndex = j + (i * W) + (k * W * H) + (l * C * W * H);
          colMajorData.push_back(initializer.float_data(rowMajorIndex));
        }
      }
    }
  }

  arma::fmat matrix(colMajorData);
  return matrix;
}

inline std::vector<double> ConvertToRowMajor(
    const arma::mat& matrix,
    const std::vector<size_t>& outputDimension)
{
  const size_t C = outputDimension[2];
  const size_t H = outputDimension[1];
  const size_t W = outputDimension[0];

  std::vector<double> returnValue;
  for (size_t i = 0; i < C; i++)
  {
    for (size_t j = 0; j < H; j++)
    {
      for (size_t k = 0; k < W; k++)
      {
        returnValue.push_back(matrix((j + (H * k) + (i * W * H)), 0));
      }
    }
  }
  return returnValue;
}

} // namespace onnx_mlpack

#endif
