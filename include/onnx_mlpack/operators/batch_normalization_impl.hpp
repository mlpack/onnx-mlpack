/**
 * @file batch_normalization_impl.hpp
 * @author Kumar Utkarsh
 *
 * Implementation of ONNX conversion to BatchNorm layer.
 */
#ifndef MLPACK_ONNX_OPERATORS_BATCH_NORMALIZATION_IMPL_HPP
#define MLPACK_ONNX_OPERATORS_BATCH_NORMALIZATION_IMPL_HPP

#include "batch_normalization.hpp"

namespace onnx_mlpack {

class BatchNorm_ : public mlpack::BatchNorm<arma::mat>
{
 public:
  inline BatchNorm_(const size_t minAxis,
                    const size_t maxAxis,
                    const double eps,
                    const bool average,
                    const double momentum,
                    onnx::GraphProto &graph,
                    const onnx::NodeProto &node) :
      BatchNorm<arma::mat>(minAxis, maxAxis, eps, average, momentum)
  {
    // setting the trained parameters
    std::string meanInput = node.input(3);
    std::string varInput = node.input(4);

    TrainingMean() = arma::conv_to<arma::mat>::from(
        ConvertToColumnMajor(Initializer(graph, meanInput)));
    TrainingVariance() = arma::conv_to<arma::mat>::from(
        ConvertToColumnMajor(Initializer(graph, varInput)));
  }
};

inline std::vector<size_t> AddBatchNormalization(
    mlpack::DAGNetwork<>& dag,
    onnx::GraphProto& graph,
    const onnx::NodeProto& node,
    std::map<std::string, double> onnxOperatorAttribute)
{
  // experiment below
  double eps = onnxOperatorAttribute["epsilon"];
  double momentum = onnxOperatorAttribute["momentum"];
  // mlpack follow width, height, channel and we want normalization only along
  // channel
  // mlpack::BatchNorm* batchNorm =
  //     new mlpack::BatchNorm(2, 2, eps, false, momentum);

  // setting the trained parameters
  // string scale_input = node.input(1); // gamma
  // string B_input = node.input(2);     // beta
  // string mean_input = node.input(3);
  // string var_input = node.input(4);

  // batchNorm->TrainingMean() = arma::conv_to<arma::mat>::from(
  //     ConvertToColumnMajor(Initializer(graph, mean_input)));
  // batchNorm->TrainingVariance() = arma::conv_to<arma::mat>::from(
  //     ConvertToColumnMajor(Initializer(graph, var_input)));

  // batchNorm->Parameters() = arma::join_cols(
  //    arma::conv_to<arma::mat>::from(
  //        ConvertToColumnMajor(Initializer(graph, meanInput))),
  //    arma::conv_to<arma::mat>::from(
  //        ConvertToColumnMajor(Initializer(graph, var_input))));
  // layerParameters.push_back(arma::join_cols(
  //    arma::conv_to<arma::mat>::from(
  //        ConvertToColumnMajor(Initializer(graph, scale_input))),
  //    arma::conv_to<arma::mat>::from(
  //        ConvertToColumnMajor(Initializer(graph, B_input)))));
  // experiment above
  size_t a = dag.Add<BatchNorm_>(2, 2, eps, false, momentum, graph, node);

  std::cout << "Added mlpack::BatchNormalization Layer" << std::endl;
  return {a};
}

inline void TransferWeightToBatchNormalization(
    mlpack::DAGNetwork<>& dag,
    std::vector<size_t>& layerIndex,
    onnx::GraphProto& graph,
    const onnx::NodeProto& node,
    std::map<std::string, double> onnxOperatorAttribute)
{
  // setting the trained parameters
  std::string scale_input = node.input(1); // gamma
  std::string B_input = node.input(2);     // beta
  // string mean_input = node.input(3);
  // string var_input = node.input(4);

  // dag.Network()[layerIndex[0]]->TrainingMean() =
  //     arma::conv_to<arma::mat>::from(
  //        ConvertToColumnMajor(Initializer(graph, mean_input)));
  // dag.Network()[layerIndex[0]]->TrainingVariance() =
  //     arma::conv_to<arma::mat>::from(
  //        ConvertToColumnMajor(Initializer(graph, var_input)));

  dag.Network()[layerIndex[0]]->Parameters() = arma::join_cols(
      arma::conv_to<arma::mat>::from(
          ConvertToColumnMajor(Initializer(graph, scale_input))),
      arma::conv_to<arma::mat>::from(
          ConvertToColumnMajor(Initializer(graph, B_input))));
}

} // namespace onnx_mlpack

#endif
