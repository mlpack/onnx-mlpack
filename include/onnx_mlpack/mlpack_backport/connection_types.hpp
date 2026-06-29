/**
 * @file onnx_mlpack/mlpack_backport/connection_types.hpp
 * @author Ryan Curtin
 *
 * Backported placement of the ConnectionTypes enum into the mlpack namespace
 * (it was not in there until after 4.8.0).
 *
 * The ONNX/mlpack converter is free software; you may redistribute it and/or
 * modify it under the terms of the 3-clause BSD license.  You should have
 * received a copy of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ONNX_MLPACK_MLPACK_BACKPORT_CONNECTION_TYPES_HPP
#define ONNX_MLPACK_MLPACK_BACKPORT_CONNECTION_TYPES_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/dag_network.hpp>

#if MLPACK_VERSION_MAJOR < 4 || \
    (MLPACK_VERSION_MAJOR == 4 && MLPACK_VERSION_MINOR < 8) || \
    (MLPACK_VERSION_MAJOR == 4 && MLPACK_VERSION_MINOR == 8 && \
     MLPACK_VERSION_PATCH == 0)
namespace mlpack {

using ConnectionTypes = ConnectionTypes;

}
#endif

#endif
