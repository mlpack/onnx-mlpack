/**
 * @file match.hpp
 * @author Ryan Curtin
 *
 * Support for matching ONNX subgraphs.  This includes all of the
 * implementations in a specific order, because the Matching class depends on
 * the Subgraph class (and vice versa).
 *
 * The ONNX/mlpack converter is free software; you may redistribute it and/or
 * modify it under the terms of the 3-clause BSD license.  You should have
 * received a copy of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ONNX_MLPACK_MATCHERS_MATCH_HPP
#define ONNX_MLPACK_MATCHERS_MATCH_HPP

#include "../extract_attribute.hpp"
#include "../extract_scalar.hpp"
#include "../extract_tensor_dims.hpp"
#include "../tensor_to_arma.hpp"

#include "matcher.hpp"
#include "subgraph.hpp"
#include "linear_no_bias_gemm.hpp"
#include "linear_no_bias_matmul.hpp"
#include "linear_gemm.hpp"
//#include "linear_gemm_add.hpp"
#include "linear_matmul_add.hpp"
#include "celu.hpp"
#include "elu.hpp"
#include "elu_piecewise.hpp"
#include "gelu.hpp"
#include "gelu_multi_op.hpp"
#include "gelu_exact.hpp"
#include "gelu_exact_multi_op.hpp"
#include "hard_sigmoid.hpp"
#include "hard_sigmoid_multi_op.hpp"
#include "hard_swish.hpp"
#include "leaky_relu.hpp"
#include "mish.hpp"
#include "mish_multi_op.hpp"
#include "prelu.hpp"
#include "prelu_multi_op.hpp"
#include "relu.hpp"
#include "selu.hpp"
#include "sigmoid.hpp"
#include "softplus.hpp"
#include "softplus_threshold.hpp"
#include "swish.hpp"
#include "tanh.hpp"
#include "max_pooling.hpp"
#include "conv.hpp"
#include "conv_add.hpp"
#include "mul_scalar.hpp"
#include "batch_norm.hpp"
#include "softmax.hpp"
#include "add_connection.hpp"
#include "mean_pooling.hpp"

#include "matcher_impl.hpp"
#include "subgraph_impl.hpp"
#include "linear_no_bias_gemm_impl.hpp"
#include "linear_no_bias_matmul_impl.hpp"
#include "linear_gemm_impl.hpp"
//#include "linear_gemm_add_impl.hpp"
#include "linear_matmul_add_impl.hpp"
#include "celu_impl.hpp"
#include "elu_impl.hpp"
#include "elu_piecewise_impl.hpp"
#include "gelu_impl.hpp"
#include "gelu_multi_op_impl.hpp"
#include "gelu_exact_impl.hpp"
#include "gelu_exact_multi_op_impl.hpp"
#include "hard_sigmoid_impl.hpp"
#include "hard_sigmoid_multi_op_impl.hpp"
#include "hard_swish_impl.hpp"
#include "leaky_relu_impl.hpp"
#include "mish_impl.hpp"
#include "mish_multi_op_impl.hpp"
#include "prelu_impl.hpp"
#include "prelu_multi_op_impl.hpp"
#include "relu_impl.hpp"
#include "selu_impl.hpp"
#include "sigmoid_impl.hpp"
#include "softplus_impl.hpp"
#include "softplus_threshold_impl.hpp"
#include "swish_impl.hpp"
#include "tanh_impl.hpp"
#include "max_pooling_impl.hpp"
#include "conv_impl.hpp"
#include "conv_add_impl.hpp"
#include "mul_scalar_impl.hpp"
#include "batch_norm_impl.hpp"
#include "softmax_impl.hpp"
#include "add_connection_impl.hpp"
#include "mean_pooling_impl.hpp"

#endif
