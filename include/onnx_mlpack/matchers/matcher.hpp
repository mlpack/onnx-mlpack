/**
 * @file matcher.hpp
 * @author Ryan Curtin
 *
 * Subgraph matcher for ONNX graphs.
 */
#ifndef ONNX_MLPACK_MATCHERS_MATCHER_HPP
#define ONNX_MLPACK_MATCHERS_MATCHER_HPP

namespace onnx_mlpack {

class Subgraph;

/**
 * A Matching is a possible matching of an ONNX graph to mlpack layers.
 *
 * It contains a DAG of mlpack layers (each one represented as a pair of mlpack
 * layer name and an index representing which of the layer's possible subgraphs
 * is being used), and each mlpack layer is associated with a series of ONNX
 * NodeProtos.
 */
struct Matching
{
  Matching(const onnx::GraphProto& g) :
      matchedNodes(g.node_size(), arma::fill::zeros) { }

  // 0/1 indicator of whether each node in the graph has been matched.
  arma::uvec matchedNodes;
  // Matchings from nodes to subgraph objects, which can be used to actually
  // convert the nodes.
  std::vector<std::pair<arma::uvec, const Subgraph*>> matches;
};

inline bool operator==(const Matching& a, const Matching& b);
inline bool Isomorphic(const Matching& a, const Matching& b);

inline std::vector<std::pair<size_t, size_t>> FindConnections(
    const Matching& m,
    const onnx::GraphProto& graph);

/**
 * Perform an exhaustive search of all possible matching of an ONNX graph.
 *
 * NOTE: the current strategy could be quite computationally expensive,
 * especially for 
 */
inline std::vector<Matching> MatchSubgraph(const onnx::GraphProto& graph);

} // namespace onnx_mlpack

#endif
