#ifndef ADD_LAYER_HPP
#define ADD_LAYER_HPP

#include "Gemm.hpp"
#include "Relu.hpp"
#include "Softmax.hpp"
#include "BatchNormalization.hpp"
#include "LeakyRelu.hpp"
#include "Conv.hpp"
#include "Mul.hpp"
#include "Add.hpp"
#include "MaxPool.hpp"
#include "GlobalAveragePool.hpp"
#include "Reshape.hpp"

#include "../model_parser/utils.hpp"

/**
 * @brief 
 */
void AddLayer(mlpack::FFN<> &ffn,
              onnx::GraphProto &graph,
              onnx::NodeProto &node,
              std::map<string, double> onnxoperatorAttribute,
              vector<arma::Mat<double>> &layerParameters);


#include "add_layer_impl.hpp"
#endif