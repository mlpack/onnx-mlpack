/**
 * @file match.hpp
 * @author Ryan Curtin
 *
 * Support for matching ONNX subgraphs.  This includes all of the
 * implementations in a specific order, because the Matching class depends on
 * the Subgraph class (and vice versa).
 */
#ifndef ONNX_MLPACK_MATCHERS_MATCH_HPP
#define ONNX_MLPACK_MATCHERS_MATCH_HPP

#include "matcher.hpp"
#include "subgraph.hpp"
#include "linear_no_bias.hpp"

#include "matcher_impl.hpp"
#include "subgraph_impl.hpp"
#include "linear_no_bias_impl.hpp"

#endif
