/**
 * @file add.hpp
 * @author Kumar Utkarsh
 *
 * ONNX "Add" operation conversion.
 */
#ifndef ONNX_MLPACK_ADD_HPP
#define ONNX_MLPACK_ADD_HPP

namespace onnx_mlpack {

inline std::vector<size_t> AddAdd(
    mlpack::DAGNetwork<>& dag,
    onnx::GraphProto& graph,
    const onnx::NodeProto& node,
    std::map<std::string, double> onnxOperatorAttribute);

} // namespace onnx_mlpack

#include "add_impl.hpp"

#endif
