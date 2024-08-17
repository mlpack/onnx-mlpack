#ifndef CONVERTER_HPP
#define CONVERTER_HPP

#include "operations/add_layer.hpp"
#include "model_parser/attribute.hpp"

/**
 * @brief Load an ONNX model from the specified file path.
 * 
 * This function reads an ONNX model from the provided file path and 
 * returns the corresponding GraphProto object.
 * 
 * @param filepath The path to the .onnx file.
 * @return onnx::GraphProto The graph representation of the ONNX model.
 */
onnx::GraphProto getGraph(const std::string& filePath);

/**
 * @brief Convert an ONNX model graph to an MLPack FFN model.
 * 
 * The core logic for converting an ONNX model into an MLPack FFN model
 * is implemented in this function. Refer to the implementation for 
 * detailed steps.
 * 
 * @param graph The ONNX model's graph representation.
 * @return mlpack::FFN<> The equivalent MLPack FFN model.
 */
mlpack::FFN<> converter(onnx::GraphProto& graph);

/**
 * @brief Utility function to validate parameter sizes.
 * 
 * This function checks whether the total number of parameters extracted 
 * from the ONNX model matches the total number required in the corresponding 
 * MLPack model.
 * 
 * @param layerParameters A vector of matrices representing the parameters 
 *                        of each layer.
 */
void printParametersSize(const std::vector<arma::Mat<double>>& layerParameters);

/**
 * @brief Flatten layer parameters for weight transfer.
 * 
 * There are two methods to transfer parameters from the ONNX model to the 
 * MLPack model:
 * 
 * Method 1: Transfer parameters one by one to each layer's `Parameters()` 
 *           function.
 * 
 * Method 2: Flatten all layer parameters and transfer them to the 
 *           `ffn.Parameters()`.
 * 
 * The converter function currently implements Method 1 for easier debugging 
 * and more intuitive weight transfer. However, Method 2 can be used by first 
 * flattening the vector of matrices and then transferring them to 
 * `ffn.Parameters()`.
 * 
 * @param layerParameters A vector of matrices representing the parameters 
 *                        of each layer.
 * @return arma::mat A flattened matrix of all layer parameters.
 */
arma::mat FlattenParameters(const std::vector<arma::Mat<double>>& layerParameters);

#include "converter_impl.hpp"

#endif // CONVERTER_HPP