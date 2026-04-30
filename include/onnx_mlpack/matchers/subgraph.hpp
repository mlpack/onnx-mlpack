/**
 * @file subgraph.hpp
 * @author Ryan Curtin
 *
 * Definition of a simple subgraph of ONNX operators (stored as strings
 * representing the operator name).
 */
#ifndef ONNX_MLPACK_MATCHERS_SUBGRAPH_HPP
#define ONNX_MLPACK_MATCHERS_SUBGRAPH_HPP

namespace onnx_mlpack {

/**
 * The Subgraph class represents a DAG of ONNX operators, represented as a
 * series of vertices and edges.
 *
 * Each vertex's data is represented as a std::string (the name of the ONNX
 * operator, e.g. "Gemm" or "Attention" or similar).
 *
 * Accessing edges can be done by interacting with the indices of each vertex.
 */
class Subgraph
{
 public:
  Subgraph() { }
  Subgraph(const std::vector<std::string>& vertices,
           const std::vector<std::pair<size_t, size_t>>& edges =
              std::vector<std::pair<size_t, size_t>>()) :
      vertices(vertices), edges(edges), numInputs(0)
  {
    // NOTE: if vertices and edges don't describe a fully-connected DAG with
    // inputs listed as the first vertices, this code is not guaranteed to work.
    std::vector<size_t> isAnyOutput(vertices.size(), false);
    outEdges.resize(vertices.size(), {});
    for (size_t i = 0; i < edges.size(); ++i)
    {
      const size_t v = edges[i].first;
      const size_t t = edges[i].second;
      outEdges.at(v).push_back(t);
      isAnyOutput[t] = true;
    }

    for (size_t i = 0; i < vertices.size(); ++i)
    {
      if (isAnyOutput[i] == true && numInputs == 0)
      {
        numInputs = i;
      }
      else if (isAnyOutput[i] == false && numInputs != 0)
      {
        throw std::invalid_argument("Subgraph::Subgraph(): input vertices must "
            "come first in the list of vertices!");
      }
    }

    // Catch edge case, where all vertices are input vertices.
    if (numInputs == 0 && isAnyOutput[0] == false)
      numInputs = vertices.size();
  }

  const size_t NumVertices() const { return vertices.size(); }
  const size_t NumEdges() const { return edges.size(); }
  const size_t NumInputs() const { return numInputs; }

  // Match the subgraph to the given ONNX graph.  This returns a list of
  // possible matchings (each individual matching is a vector of nodes).
  inline std::vector<Matching> Match(const onnx::GraphProto& graph,
                                     const Matching& parentMatch) const;

  inline void UpdateMatchedNodes(size_t i, std::vector<bool>& matched) const;

  inline std::vector<arma::uvec> MatchSubDAG(
      const size_t i,
      const arma::uvec& currentMatching,
      const std::vector<bool>& matchedNodes,
      const onnx::GraphProto& graph,
      const std::unordered_map<std::string, std::vector<size_t>>& nodeMap,
      const std::vector<size_t>& possibleGraphNodes) const;

  virtual bool Validate(const arma::uvec& indices,
                        const onnx::GraphProto& graph) const = 0;

  virtual void Convert(const arma::uvec& indices,
                       const onnx::GraphProto& graph,
                       mlpack::DAGNetwork<>& network) const = 0;

  // This is not needed by every subgraph, and can be left un-overloaded if
  // the layer has no weights.
  virtual void TransferWeights(const arma::uvec& indices,
                               const onnx::GraphProto& graph,
                               mlpack::Layer<>* layer) const { }

 private:
  std::vector<std::string> vertices;
  std::vector<std::pair<size_t, size_t>> edges;
  size_t numInputs;
  std::vector<std::vector<size_t>> outEdges;
  std::vector<std::unordered_set<size_t>> inEdges;
};

} // namespace onnx_mlpack

#endif
