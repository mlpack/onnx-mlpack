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
#include "linear_no_bias_gemm.hpp"
#include "linear_no_bias_matmul.hpp"
#include "linear_gemm.hpp"
//#include "linear_gemm_add.hpp"
#include "linear_matmul_add.hpp"
#include "celu.hpp"
#include "elu.hpp"
#include "gelu.hpp"
#include "gelu_exact.hpp"
#include "hard_sigmoid.hpp"
#include "hard_swish.hpp"
#include "leaky_relu.hpp"
#include "mish.hpp"
#include "mish_multi_op.hpp"
#include "prelu.hpp"
#include "relu.hpp"
#include "selu.hpp"
#include "sigmoid.hpp"
#include "softplus.hpp"
#include "softplus_threshold.hpp"
#include "swish.hpp"
#include "tanh.hpp"

#include "matcher_impl.hpp"
#include "subgraph_impl.hpp"
#include "linear_no_bias_gemm_impl.hpp"
#include "linear_no_bias_matmul_impl.hpp"
#include "linear_gemm_impl.hpp"
//#include "linear_gemm_add_impl.hpp"
#include "linear_matmul_add_impl.hpp"
#include "celu_impl.hpp"
#include "elu_impl.hpp"
#include "gelu_impl.hpp"
#include "gelu_exact_impl.hpp"
#include "hard_sigmoid_impl.hpp"
#include "hard_swish_impl.hpp"
#include "leaky_relu_impl.hpp"
#include "mish_impl.hpp"
#include "mish_multi_op_impl.hpp"
#include "prelu_impl.hpp"
#include "relu_impl.hpp"
#include "selu_impl.hpp"
#include "sigmoid_impl.hpp"
#include "softplus_impl.hpp"
#include "softplus_threshold_impl.hpp"
#include "swish_impl.hpp"
#include "tanh_impl.hpp"

#endif
