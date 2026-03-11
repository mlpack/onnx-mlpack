/**
 * @file subgraph_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of Subgraph class and its matching functionality.
 */
#ifndef ONNX_MLPACK_MATCHERS_SUBGRAPH_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_SUBGRAPH_IMPL_HPP

namespace onnx_mlpack {

inline std::vector<Matching> Subgraph::Match(
    const onnx::GraphProto& graph,
    const Matching& parentMatch) const
{
  std::vector<Matching> results;

  // If this subgraph is empty, do nothing.
  if (vertices.size() == 0)
  {
    return results;
  }

  // Iterate over each of the nodes in the graph, and try to use that as the
  // root of a potential subgraph.
  for (size_t n = 0; n < graph.node_size(); ++n)
  {
    if (parentMatch.matchedNodes[n] == 1)
      continue; // This can't be a match---it's already matched.

    const onnx::NodeProto& node = graph.node(n);
    if (node.op_type() == vertices[0])
    {
      // This is a possible root for the subgraph.

      // TODO: actually match the subgraph!

      arma::uvec indices({ n });
      if (this->Validate(indices, graph))
      {
        Matching m(parentMatch);
        m.matchedNodes[n] = 1;
        m.matches.push_back(std::make_pair<arma::uvec, const Subgraph*>(
            std::move(indices), this));
        results.push_back(std::move(m));
      }
    }
  }

  return results;
}

} // namespace onnx_mlpack

#endif
