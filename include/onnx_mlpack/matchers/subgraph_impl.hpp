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

  // Build a mapping from ONNX node types to a list of nodes in the graph, for
  // easy lookup.
  std::unordered_map<std::string, std::vector<size_t>> nodeMap;
  for (size_t n = 0; n < graph.node_size(); ++n)
  {
    const std::string& op = graph.node(n).op_type();
    if (nodeMap.count(op) == 0)
      nodeMap[op] = std::vector<size_t>();

    nodeMap[op].push_back(n);
  }

  // General matching strategy:
  //
  // For each input node in the subgraph, we will match the entire sub-DAG
  // rooted at that input node.  There can be multiple matches of this sub-DAG.
  // For each match we find, we then proceed to the next input node in the
  // subgraph and match the parts of its sub-DAG that have not already been
  // matched.

  std::vector<bool> matchedNodes(graph.node_size(), false);
  arma::uvec parentMatching = (vertices.size() + 1) * arma::ones<arma::uvec>(
      graph.node_size());
  for (size_t i = 0; i < parentMatch.matchedNodes.n_elem; ++i)
    if (parentMatch.matchedNodes[i] == 1)
      parentMatching[i] = vertices.size();

  // Match all input nodes to a set of candidates.  The arma::uvec maps an input
  // vertex to the graph node it has been matched to.
  std::stack<arma::uvec> matchStack;
  matchStack.push(parentMatching);
  for (size_t i = 0; i < numInputs; ++i)
  {
    // This will hold all of the elements on the stack that have been further
    // matched to input node i.
    std::vector<arma::uvec> outputMatches;
    while (!matchStack.empty())
    {
      const arma::uvec& matching = matchStack.top();
      // Try to match to any nodes of the right type.  If we have none, we can
      // terminate early.
      if (nodeMap.count(vertices[i]) > 0)
      {
        std::vector<size_t> possibleGraphNodes;
        for (size_t j = 0; j < nodeMap[vertices[i]].size(); ++j)
          if (matching[nodeMap[vertices[i]][j]] == vertices.size() + 1)
            possibleGraphNodes.push_back(nodeMap[vertices[i]][j]);

        std::vector<arma::uvec> matches = MatchSubDAG(i, matching, matchedNodes,
            graph, nodeMap, possibleGraphNodes);
        for (size_t k = 0; k < matches.size(); ++k)
          outputMatches.push_back(std::move(matches[k]));
      }

      matchStack.pop();
    }

    for (size_t j = 0; j < outputMatches.size(); ++j)
      matchStack.push(std::move(outputMatches[j]));

    // Since we have recursed the DAG for input node i, we are guaranteed that
    // anything in the stack has those sub-DAG nodes matched, so we can update
    // matchedNodes.
    UpdateMatchedNodes(i, matchedNodes);
  }

  // Process the results into Matching objects.
  while (!matchStack.empty())
  {
    const arma::uvec& matching = matchStack.top();
    arma::uvec subgraphIndices(vertices.size());
    for (size_t i = 0; i < matching.n_elem; ++i)
      if (matching[i] < vertices.size())
        subgraphIndices[matching[i]] = i;

    // Before accepting, first validate.
    if (this->Validate(subgraphIndices, graph))
    {
      results.push_back(parentMatch);
      results.back().matchedNodes = (matching != (vertices.size() + 1));
      results.back().matches.push_back(
          std::make_pair(std::move(subgraphIndices), this));
    }

    matchStack.pop();
  }

  return results;
}

// Any nodes reachable in the DAG from node i will have their values in
// `matched` set to true.
inline void Subgraph::UpdateMatchedNodes(size_t i, std::vector<bool>& matched)
    const
{
  matched[i] = true;
  for (const size_t& j : outEdges[i])
    UpdateMatchedNodes(j, matched);
}

inline std::vector<arma::uvec> Subgraph::MatchSubDAG(
    const size_t i,
    const arma::uvec& currentMatching,
    const std::vector<bool>& matchedNodes,
    const onnx::GraphProto& graph,
    const std::unordered_map<std::string, std::vector<size_t>>& nodeMap,
    const std::vector<size_t>& possibleGraphNodes) const
{
  std::vector<arma::uvec> matchings;
  if (matchedNodes[i])
  {
    // This node has already been matched by another branch of the DAG, so all
    // we need to do is ensure that the matching we have is using one of the
    // possible graph nodes that i can be matched to.
    for (const size_t& n : possibleGraphNodes)
    {
      if (currentMatching[n] == i)
      {
        matchings.push_back(currentMatching);
        return matchings;
      }
    }
  }

  // Try to match node i to a node in the graph.  We assume that
  // possibleGraphNodes only contains nodes of the right operation type that
  // have not been matched already.
  for (const size_t& n : possibleGraphNodes)
  {
    std::vector<arma::uvec> nodeMatchings;

    // Match vertex i to node n.
    arma::uvec matching = currentMatching;
    matching[n] = i;
    nodeMatchings.push_back(std::move(matching));

    // This will hold all child matchings based on this matching.
    for (const size_t j : outEdges[i])
    {
      std::vector<arma::uvec> edgeMatchings;
      // If there are no nodes with the right operation in the graph, we can
      // terminate early.
      if (nodeMap.count(vertices[j]) == 0)
        return matchings; // This will be empty.

      // Try this with all current matchings.
      for (const arma::uvec& m : nodeMatchings)
      {
        // If the out-edge is already matched, don't recurse.
        if (arma::any(m == j))
        {
          edgeMatchings.push_back(m);
          continue;
        }

        // Collect the set of possible nodes that could be children.
        std::vector<size_t> childPossibleGraphNodes;
        for (const size_t& k : nodeMap.at(vertices[j]))
        {
          // Note we allow both unmatched nodes, and nodes where the out-edge is
          // already successfully matched.
          if (m[k] == vertices.size() + 1 || (matchedNodes[j] && m[k] == j))
          {
            // ONNX graph node k is not matched; is it connected to an output of
            // node i?  TODO: this should be cleaned up and precomputed, or put
            // somewhere else.
            bool isOutput = false;
            for (size_t l = 0; l < graph.node(n).output_size(); ++l)
            {
              bool found = false;
              for (size_t kk = 0; kk < graph.node(k).input_size(); ++kk)
              {
                if (graph.node(k).input(kk) == graph.node(n).output(l))
                {
                  found = true;
                  break;
                }
              }

              if (found)
                childPossibleGraphNodes.push_back(k);
            }
          }
        }

        std::vector<arma::uvec> results = MatchSubDAG(j, m, matchedNodes, graph,
            nodeMap, childPossibleGraphNodes);

        for (size_t l = 0; l < results.size(); ++l)
          edgeMatchings.push_back(std::move(results[l]));
      }

      // Update the set of matchings and proceed to the next out-edge.
      nodeMatchings = std::move(edgeMatchings);
    }

    // Add all valid matchings for this node to the final result.
    for (size_t k = 0; k < nodeMatchings.size(); ++k)
      matchings.push_back(std::move(nodeMatchings[k]));
  }

  return matchings;
}

} // namespace onnx_mlpack

#endif
