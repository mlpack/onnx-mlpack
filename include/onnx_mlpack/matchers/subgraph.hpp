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
      vertices(vertices), edges(edges)
  {
    // NOTE: if vertices and edges don't describe a fully-connected DAG with one
    // input and the first vertex being that first input, this code is not
    // guaranteed to work.
    outEdges.resize(vertices.size(), {});
    for (size_t i = 0; i < edges.size(); ++i)
    {
      const size_t v = edges[i].first;
      const size_t t = edges[i].second;
      outEdges.at(v).push_back(t);
    }
  }

  const size_t NumVertices() const { return vertices.size(); }
  const size_t NumEdges() const { return edges.size(); }

  // Match the subgraph to the given ONNX graph.  This returns a list of
  // possible matchings (each individual matching is a vector of nodes).
  inline std::vector<Matching> Match(const onnx::GraphProto& graph,
                                     const Matching& parentMatch) const;

  inline std::vector<arma::uvec> MatchNode(
      const size_t v,
      const size_t n,
      const onnx::GraphProto& graph,
      const arma::uvec& currentMatching = arma::uvec()) const;

  virtual bool Validate(const arma::uvec& indices,
                        const onnx::GraphProto& graph) const = 0;

  virtual void Convert(const arma::uvec& indices,
                       const onnx::GraphProto& graph,
                       mlpack::DAGNetwork<>& network) const = 0;

  virtual void TransferWeights(const arma::uvec& indices,
                               const onnx::GraphProto& graph,
                               mlpack::Layer<>* layer) const = 0;

 private:
  std::vector<std::string> vertices;
  std::vector<std::pair<size_t, size_t>> edges;
  std::vector<std::vector<size_t>> outEdges;
};

} // namespace onnx_mlpack

#endif
