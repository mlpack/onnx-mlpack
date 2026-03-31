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

    // Try to match the subgraph to the root of the subgraph.
    std::vector<arma::uvec> matches = MatchNode(0, n, graph);
    if (matches.size() > 0)
    {
      for (size_t i = 0; i < matches.size(); ++i)
      {
        // Validate the subgraph against any extra restrictions of the layer.
        if (this->Validate(matches[i], graph))
        {
          Matching m(parentMatch);
          for (size_t j = 0; j < matches[i].n_elem; ++j)
            m.matchedNodes[matches[i][j]] = 1;
          m.matches.push_back(std::make_pair<arma::uvec, const Subgraph*>(
              std::move(matches[i]), this));
          results.push_back(std::move(m));
        }
      }
    }
  }

  return results;
}

inline std::vector<arma::uvec> Subgraph::MatchNode(
    const size_t v,
    const size_t n,
    const onnx::GraphProto& graph,
    const arma::uvec& currentMatching) const
{
  std::vector<arma::uvec> result;

  if (currentMatching.n_elem != 0 && currentMatching[v] != graph.node_size())
    return result; // This node is already matched.

  // Check that this node matches the vertex.
  const onnx::NodeProto& node = graph.node(n);
  if (node.op_type() == vertices[v])
  {
    // If this is not the end of the DAG, we need to make sure the out degree
    // matches what we are looking for.
    if (outEdges[v].size() != 0 &&
        node.output_size() != outEdges[v].size())
    {
      return result;
    }

    result.push_back(currentMatching);
    // Initialize the matching if it is empty.
    if (result.back().n_elem == 0)
    {
      result.back() = graph.node_size() *
          arma::ones<arma::uvec>(vertices.size());
    }
    result.back()[v] = n;

    // Compute matchings on each child, maintaining a list of matchings that are
    // still valid.
    for (size_t i = 0; i < outEdges[v].size(); ++i)
    {
      std::vector<arma::uvec> updatedCandidates;
      for (size_t m = 0; m < result.size(); ++m)
      {
        for (size_t k = 0; k < node.output_size(); ++k)
        {
          // Longer-term TODO: replace this search with a precomputation of the
          // ONNX graph's network structure.
          const std::string& outTensorName = node.output(k);
          size_t outIndex = graph.node_size();
          for (size_t kk = 0; kk < graph.node_size(); ++kk)
          {
            bool hasInput = false;
            for (size_t ll = 0; ll < graph.node(kk).input_size(); ++ll)
            {
              if (graph.node(kk).input(ll) == outTensorName)
              {
                hasInput = true;
                break;
              }
            }

            if (!hasInput)
              continue;

            // Only try with node kk if it isn't already assigned.
            if (result[m][outEdges[v][i]] == graph.node_size())
            {
              std::vector<arma::uvec> childMatch = MatchNode(outEdges[v][i], kk,
                  graph, result[m]);
              updatedCandidates.insert(updatedCandidates.end(),
                  childMatch.begin(), childMatch.end());
            }
          }
        }
      }

      // Now we can replace our results with the updated matches.
      result = std::move(updatedCandidates);
      if (result.size() == 0)
        break; // Shortcut... nothing matched, so we can't match the rest.
    }
  }

  return result;
}

} // namespace onnx_mlpack

#endif
