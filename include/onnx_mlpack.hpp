/**
 * @file onnx_mlpack.hpp
 *
 * Include this file to include all the necessary functions for the ONNX-mlpack
 * converter.
 */
#ifndef ONNX_MLPACK_HPP
#define ONNX_MLPACK_HPP

// Prerequisites.
#include <stdint.h>
#include <mlpack.hpp>

// If the mlpack version is too old, we need to include the backported Scale
// layer.
#if MLPACK_VERSION_MAJOR <= 4 || \
    (MLPACK_VERSION_MAJOR == 4 && MLPACK_VERSION_MINOR <= 8)
  #include "onnx_mlpack/mlpack_backport/scale.hpp"
#endif

#include "onnx_mlpack/convert.hpp"

#endif
