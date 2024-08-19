#ifndef ATTRIBUTE_HPP
#define ATTRIBUTE_HPP

#include <iostream>
#include "helper.hpp"
#include "utils.hpp"
#include "onnx_pb.h"

 /**
 * @brief Extracts and maps the attributes associated with an ONNX node to a form 
 *        that can be used in MLPack layers.
 *
 * This function extracts relevant attributes from an ONNX node within a graph 
 * and organizes them into a map where the keys are attribute names (as strings) 
 * and the values are their corresponding numeric values (as doubles).
 *
 * @param graph The ONNX graph containing the node.
 * @param node The ONNX node for which attributes need to be extracted.
 * @return A map where each key-value pair represents an attribute name and its value.
 * 
 * @note Example output for a Convolution node:
 * @code
 * {
 *   {"paddingType", 0.0},
 *   {"auto_pad", 0.0},
 *   {"pad_top", 0.0},
 *   {"pad_bottom", 0.0},
 *   {"pad_right", 0.0},
 *   {"pad_left", 0.0},
 *   {"group", 1.0},
 *   {"dilation_height", 1.0},
 *   {"dilation_width", 1.0},
 *   {"kernel_height", 1.0},
 *   {"kernel_width", 1.0},
 *   {"stride_height", 1.0},
 *   {"stride_width", 1.0}
 * }
 * @endcode
 */
std::map<std::string, double> OnnxOperatorAttribute(onnx::GraphProto& graph, onnx::NodeProto& node);

#include "attribute_impl.hpp"

#endif // ATTRIBUTE_HPP
