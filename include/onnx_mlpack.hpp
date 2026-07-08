/**
 * @file onnx_mlpack.hpp
 *
 * Include this file to include all the necessary functions for the ONNX-mlpack
 * converter.
 *
 * The ONNX/mlpack converter is free software; you may redistribute it and/or
 * modify it under the terms of the 3-clause BSD license.  You should have
 * received a copy of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ONNX_MLPACK_HPP
#define ONNX_MLPACK_HPP

// Prerequisites.
#include <stdint.h>
#include <mlpack.hpp>

// Backport required pieces from newer versions of mlpack.
#include "onnx_mlpack/mlpack_backport/scale.hpp"
#include "onnx_mlpack/mlpack_backport/connection_types.hpp"

#include "onnx_mlpack/convert.hpp"

#endif
