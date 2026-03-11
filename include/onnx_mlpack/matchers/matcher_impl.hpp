/**
 * @file matcher_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of subgraph matcher for ONNX graphs.
 */
#ifndef ONNX_MLPACK_MATCHERS_MATCHER_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_MATCHER_IMPL_HPP

#include "matcher.hpp"
#include "subgraph.hpp"

namespace onnx_mlpack {

inline bool operator==(const Matching& a, const Matching& b)
{
  if (a.matchedNodes.n_elem != b.matchedNodes.n_elem)
    return false;

  if (arma::any(a.matchedNodes != b.matchedNodes))
    return false;

  if (a.matches.size() != b.matches.size())
    return false;

  for (size_t i = 0; i < a.matches.size(); ++i)
  {
    if (a.matches[i].first.n_elem != b.matches[i].first.n_elem)
      return false;
    if (any(a.matches[i].first != b.matches[i].first))
      return false;
    if (&(a.matches[i].second) != &(b.matches[i].second))
      return false;
  }

  return true;
}

struct MatchingLess
{
  bool operator()(const std::pair<arma::uvec, const Subgraph*>& a,
                  const std::pair<arma::uvec, const Subgraph*>& b) const
  {
    // Sorting is done entirely on the first element of the arma::uvec.
    // This works because an individual matching will not have any duplicate
    // elements in any arma::uvecs.
    if (a.first.n_elem == 0)
      return true; // This case should never happen...
    if (b.first.n_elem == 0)
      return false;
    return (a.first[0] < b.first[0]);
  }
};

inline bool Isomorphic(const Matching& a, const Matching& b)
{
  // Cannot be isomorphic if the matched nodes differ in any way.
  if (a.matchedNodes.n_elem != b.matchedNodes.n_elem)
    return false;

  if (arma::any(a.matchedNodes != b.matchedNodes))
    return false;

  if (a.matches.size() != b.matches.size())
    return false;

  // To check for isomorphic matchings, we just need to sort `matches`.
  std::vector<std::pair<arma::uvec, const Subgraph*>> matchesA(a.matches);
  std::vector<std::pair<arma::uvec, const Subgraph*>> matchesB(b.matches);
  std::sort(matchesA.begin(), matchesA.end(), MatchingLess());
  std::sort(matchesB.begin(), matchesB.end(), MatchingLess());

  for (size_t i = 0; i < matchesA.size(); ++i)
  {
    if (matchesA[i].first.n_elem != matchesB[i].first.n_elem)
      return false;
    if (any(matchesA[i].first != matchesB[i].first))
      return false;
    if (&(matchesA[i].second) != &(matchesB[i].second))
      return false;
  }

  return true;
}

inline std::vector<std::pair<size_t, size_t>> FindConnections(
    const Matching& m,
    const onnx::GraphProto& graph)
{
  std::vector<std::pair<size_t, size_t>> result;

  // Construct a mapping from ONNX layer IDs to matching ID.
  std::map<size_t, size_t> layerMap;
  for (size_t i = 0; i < m.matches.size(); ++i)
    for (size_t j = 0; j < m.matches[i].first.n_elem; ++j)
      layerMap[m.matches[i].first[j]] = i;

  for (size_t i = 0; i < m.matches.size(); ++i)
  {
    // Find the ONNX operation outputs of each ONNX operation in the subgraph.
    for (size_t j = 0; j < m.matches[i].first.n_elem; ++j)
    {
      const onnx::NodeProto& n = graph.node(m.matches[i].first[j]);
      for (size_t k = 0; k < n.output_size(); ++k)
      {
        const std::string& outputName = n.output(k);
        // Search for nodes that use the output tensor as input.
        for (size_t l = 0; l < graph.node_size(); ++l)
        {
          const onnx::NodeProto& n2 = graph.node(l);
          for (size_t k2 = 0; k2 < n2.input_size(); ++k2)
          {
            if (n2.input(k2) == outputName)
              result.push_back(std::make_pair(i, layerMap[l]));
          }
        }
      }
    }
  }

  return result;
}

inline Matching Matcher(const onnx::GraphProto& graph)
{
  // First collect all of the subgraphs that we might be trying to match.
  std::vector<Subgraph*> subgraphs;

  std::vector<Subgraph*> s = LinearNoBiasSubgraph::Subgraphs();
  subgraphs.insert(subgraphs.end(), s.begin(), s.end());

  // TODO: support more than one matching operation...
  assert(subgraphs.size() == 1);

  // Now iterate over all the ONNX nodes to try and match.
  std::stack<Matching> matchStack;
  matchStack.push(Matching(graph));

  std::vector<Matching> fullMatchings;

  // General algorithmic idea:
  //
  // 1. With the current state of the graph, perform as many possible matchings
  // as we can.
  //
  // 2. Any matchings that are complete (e.g. all ONNX graph nodes are matched)
  // can be added to the result set.
  //
  // 3. With each possible incomplete matching, recurse and perform more
  // matchings.
  //
  // Once all possible matchings are collected (this is a tree, but we will just
  // hold each leaf in the tree as a standalone object), then we need to find
  // which matchings are the best.
  //
  // First we need to filter out isomorphic mappings.  That should be easy
  // enough.
  //
  // Then, we need to rank the mappings by how good they are.  For starters, we
  // can just use the number of mlpack layers that are in the matching.

  while (matchStack.size() > 0)
  {
    // Match each subgraph to the current state.
    std::vector<Matching> matchings;
    for (size_t s = 0; s < subgraphs.size(); ++s)
    {
      std::vector<Matching> subMatchings = subgraphs[s]->Match(graph,
          matchStack.top());
      matchings.insert(matchings.end(), subMatchings.begin(),
          subMatchings.end());
    }
    matchStack.pop();

    // Look through each matching we received.
    for (size_t m = 0; m < matchings.size(); ++m)
    {
      // Is the matching complete?
      if (arma::all(matchings[m].matchedNodes == 1))
      {
        fullMatchings.push_back(std::move(matchings[m]));
      }
      else
      {
        // The matching is not complete yet.  We have to recurse with the
        // current state of the matching.
        matchStack.push(std::move(matchings[m]));
      }
    }
  }

  if (fullMatchings.size() == 0)
  {
    throw std::runtime_error("No complete matching found!");
  }

  // Now we need to filter out isomorphic mappings.
  arma::uvec keep(fullMatchings.size(), arma::fill::ones);
  for (size_t i = 0; i < fullMatchings.size(); ++i)
  {
    for (size_t j = 1; j < fullMatchings.size(); ++j)
    {
      if (keep[i] == 0)
        continue;

      if (Isomorphic(fullMatchings[j], fullMatchings[i]))
      {
        keep[j] = 0;
        break;
      }
    }
  }

  std::vector<Matching> finalMatchings;
  for (size_t i = 0; i < fullMatchings.size(); ++i)
    if (keep[i])
      finalMatchings.push_back(std::move(fullMatchings[i]));

  // Lastly, we need to rank the mappings.
  // As a very simple first heuristic, we will just count the number of layers
  // and minimize.
  arma::uvec numLayers(finalMatchings.size());
  for (size_t i = 0; i < finalMatchings.size(); ++i)
    numLayers[i] = finalMatchings[i].matches.size();

  size_t bestMatch = numLayers.index_min();

  // Clean up.
  for (size_t s = 0; s < subgraphs.size(); ++s)
  {
    delete subgraphs[s];
    subgraphs[s] = nullptr;
  }

  return finalMatchings[bestMatch];
}

} // namespace onnx_mlpack

#endif
