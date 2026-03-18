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
  Subgraph(const std::vector<std::string>& vertices) : vertices(vertices) { }

  const size_t NumVertices() const { return vertices.size(); }
  const size_t NumEdges() const { return edges.size(); }

  // Match the subgraph to the given ONNX graph.  This returns a list of
  // possible matchings (each individual matching is a vector of nodes).
  inline std::vector<Matching> Match(const onnx::GraphProto& graph,
                                     const Matching& parentMatch) const;

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
};

} // namespace onnx_mlpack

#endif
