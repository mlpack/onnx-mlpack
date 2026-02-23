/**
 * @file gemm_impl.hpp
 * @author Kumar Utkarsh
 *
 * Implementation of ONNX GEMM operation conversion.
 */
#ifndef ONNX_MLPACK_OPERATIONS_GEMM_IMPL_HPP
#define ONNX_MLPACK_OPERATIONS_GEMM_IMPL_HPP

#include "gemm.hpp"

namespace onnx_mlpack {

inline std::vector<size_t> AddGemm(
    mlpack::DAGNetwork<>& dag,
    onnx::GraphProto graph,
    onnx::NodeProto node,
    std::map<std::string, double> onnxOperatorAttribute)
{
  // mlpack::LinearNoBias *linearNoBias = new mlpack::LinearNoBias(
  //    FindOutputDimension(graph, node));
  // mlpack::Add *add = new mlpack::Add();

  size_t a = dag.Add<mlpack::LinearNoBias>(FindOutputDimension(graph, node));
  size_t b = dag.Add<mlpack::Add>();
  dag.Connect(a, b);

  std::cout << "Added Gemm Layer" << std::endl;
  return {a, b};

  // getting the weights in correct dimension
  // arma::mat weights = onnxOperatorAttribute["alpha"] *
  //    ExtractWeights(graph, node, onnxOperatorAttribute["transB"]);

  // layerParameters.push_back(weights);
  // std::cout << "Added mlpack::LinearNoBias Layer" << std::endl;

  // mlpack::Add *add = new mlpack::Add();
  // // getting the biases in correct dimension
  // arma::mat biases = ExtractBiases(graph, node);
  // // biases.print("biases");
  // layerParameters.push_back(biases);

  // mlpack::MultiLayer* multiLayer = new mlpack::MultiLayer();

  // return dag.Add(add);
  // std::cout << "Added mlpack::Add Layer" << std::endl;
}

inline void TransferWeightToGemm(
    mlpack::DAGNetwork<>& dag,
    std::vector<size_t>& layerIndex,
    onnx::GraphProto& graph,
    const onnx::NodeProto& node,
    std::map<std::string, double> onnxOperatorAttribute)
{
  // gemm layer will be a multilayer with LinearNoBias and Add layer
  dag.Network()[layerIndex[0]]->Parameters() = ExtractWeights(graph, node,
      onnxOperatorAttribute["transB"]);

  dag.Network()[layerIndex[1]]->Parameters() = ExtractBiases(graph, node);
}

inline size_t FindOutputDimension(onnx::GraphProto graph, onnx::NodeProto node)
{
  // 3rd input name of Gemm node points to onnx Add operator initializer
  std::string addInitializerName = node.input(2);
  for (onnx::TensorProto initializer : graph.initializer())
  {
    if (initializer.name() == addInitializerName)
    {
      return initializer.dims(0);
    }
  }
  throw std::runtime_error("No initializer for the third input of ONNX Gemm "
      "node found!");
}

inline arma::mat ExtractWeights(onnx::GraphProto graph,
                                onnx::NodeProto node,
                                bool transposed)
{
  // finding the initializer in which the weights are stored
  std::string inputName = node.input(1);
  onnx::TensorProto weightInitializer;
  for (auto initializer : graph.initializer())
    if (initializer.name() == inputName)
      weightInitializer = initializer;

  if (weightInitializer.data_type() == onnx::TensorProto::FLOAT)
  {
    std::vector<float> tensorData(weightInitializer.raw_data().size() /
        sizeof(float));
    memcpy(tensorData.data(), weightInitializer.raw_data().data(),
        weightInitializer.raw_data().size());

    // dimension of matrix stored in initializer
    size_t rows = weightInitializer.dims(0);
    size_t cols = weightInitializer.dims(1);
    arma::fvec armaVector(tensorData); // vector form
    // pytorch works on row major format and armadillo works on column major
    // format
    // this is how the weights will looks in pytorch if we print it
    arma::fmat fWeights = arma::reshape(armaVector, cols, rows).t();
    arma::mat weights = arma::conv_to<arma::mat>::from(fWeights);

    return transposed ? weights : weights.t();
  }

  throw std::runtime_error("error occured at weight extraction in gemm");
}

inline arma::mat ExtractBiases(onnx::GraphProto graph, onnx::NodeProto node)
{
  // finding the initializer in which biases are stored
  std::string input_name = node.input(2);
  onnx::TensorProto biasInitializer;

  for (auto initializer : graph.initializer())
    if (initializer.name() == input_name)
      biasInitializer = initializer;

  if (biasInitializer.data_type() == onnx::TensorProto::FLOAT)
  {
    std::vector<float> tensorData(biasInitializer.raw_data().size() /
        sizeof(float));
    memcpy(tensorData.data(), biasInitializer.raw_data().data(),
        biasInitializer.raw_data().size());

    // dimension of matrix stored in initializer
    size_t elements = biasInitializer.dims(0);
    arma::fvec armaVector(tensorData);
    arma::fmat fBiases = arma::reshape(armaVector, 1, elements);
    arma::mat biases = arma::conv_to<arma::mat>::from(fBiases);

    return biases;
  }

  throw std::runtime_error("error occured at bias extraction in gemm");
}

} // namespace onnx_mlpack

#endif
