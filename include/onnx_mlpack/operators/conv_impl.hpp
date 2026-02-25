/**
 * @file conv_impl.hpp
 * @author Kumar Utkarsh
 *
 * Implementation of ONNX Conv operation conversion.
 */
#ifndef ONNX_MLPACK_OPERATORS_CONV_IMPL_HPP
#define ONNX_MLPACK_OPERATORS_CONV_IMPL_HPP

#include "conv.hpp"

namespace onnx_mlpack {

inline std::vector<size_t> AddConv(
    mlpack::DAGNetwork<> &dag,
    onnx::GraphProto graph,
    onnx::NodeProto node,
    std::map<std::string, double> onnxOperatorAttribute)
{
  bool useBias = false;
  if (node.input().size() == 3)
    useBias = true;
  onnx::TensorProto initializerWeights;
  std::string initializerWeightsName = node.input(1);
  initializerWeights = Initializer(graph, initializerWeightsName);

  // if (node.input().size() == 2)
  // {
  //   string initializerWeightsName = node.input(1);
  //   useBias = false;
  //   initializerWeights = Initializer(graph, initializerWeightsName);
  //   layerParameters.push_back(arma::conv_to<arma::mat>::from(
  //       ConvertToColumnMajor(initializerWeights)));
  // }
  // else if (node.input().size() == 3)
  // {
  //   std::string initializerWeightsName = node.input(1);
  //   std::string initializerBiasName = node.input(2);
  //   useBias = true;
  //   initializerWeights = Initializer(graph, initializerWeightsName);
  //   initializerBias = Initializer(graph, initializerBiasName);
  //   layerParameters.push_back(arma::join_cols(
  //       arma::conv_to<arma::mat>::from(
  //          ConvertToColumnMajor(initializerWeights)),
  //       arma::conv_to<arma::mat>::from(
  //          ConvertToColumnMajor(initializerBias))));
  // }

  // converting the onnx attribute to mlpack layer parameters
  // size_t maps = FindConvMap(ffn, graph, node);
  size_t maps = initializerWeights.dims(0);
  size_t kernelHeight = onnxOperatorAttribute["kernel_height"];
  size_t kernelWidth = onnxOperatorAttribute["kernel_width"];
  size_t strideHeight = onnxOperatorAttribute["stride_height"];
  size_t strideWidth = onnxOperatorAttribute["stride_width"];
  size_t group = onnxOperatorAttribute["group"];
  size_t padW = 0;
  size_t padH = 0;
  std::string paddingType = "none";
  if (onnxOperatorAttribute["auto_pad_or_pads"] == 0) // auto_pad
  {
    if (onnxOperatorAttribute["auto_pad"] == 0)
    {
      // NOT_SET => explicit value will be used
      paddingType = "none";
    }
    if (onnxOperatorAttribute["auto_pad"] == 1 ||
        onnxOperatorAttribute["auto_pad"] == 1) // SAME_UPPER OR SAME_LOWER
    {
      paddingType = "same";
    }
    if (onnxOperatorAttribute["auto_pad"] == 3) // VALID
    {
      paddingType = "valid";
    }
  }
  else if (onnxOperatorAttribute["auto_pad_or_pads"] == 1) // pads
  {
    padW = (onnxOperatorAttribute["pad_right"] +
        onnxOperatorAttribute["pad_left"]) / 2;
    padH = (onnxOperatorAttribute["pad_top"] +
        onnxOperatorAttribute["pad_bottom"]) / 2;
  }

  std::vector<size_t> v;
  size_t a = dag.Add<mlpack::GroupedConvolution>(maps, kernelWidth,
      kernelHeight, group, strideWidth, strideHeight, padW, padH, paddingType,
      useBias);
  std::cout << "added grouped conv" << std::endl;

  // if (group == 1)
  // {
  //   size_t a = dag.Add<mlpack::Convolution>(maps, kernelWidth, kernelHeight,
  //       strideWidth, strideHeight, padW, padH, paddingType, useBias);
  //   std::cout << "Added mlpack::Conv Layer" << std::endl;
  //   v.push_back(a);
  // }
  // else
  // {
  //   size_t a = dag.Add<mlpack::Convolution>(maps, kernelWidth, kernelHeight,
  //       group, strideWidth, strideHeight, padW, padH, paddingType, useBias);
  //   cout << "Added mlpack::GroupedConv Layer" << endl;
  //   v.push_back(a);
  // }

  return {a};
}

void TransferWeightToConv(mlpack::DAGNetwork<>& dag,
                          std::vector<size_t>& layerIndex,
                          onnx::GraphProto& graph,
                          const onnx::NodeProto& node,
                          std::map<std::string, double> onnxOperatorAttribute)
{
  bool useBias = false;
  onnx::TensorProto initializerWeights;
  onnx::TensorProto initializerBias;

  if (node.input().size() == 2)
  {
    std::string initializerWeightsName = node.input(1);
    useBias = false;
    initializerWeights = Initializer(graph, initializerWeightsName);
    // layerParameters.push_back(arma::conv_to<arma::mat>::from(
    //    ConvertToColumnMajor(initializerWeights)));
    dag.Network()[layerIndex[0]]->Parameters() = arma::conv_to<arma::mat>::from(
        ConvertToColumnMajor(initializerWeights));
  }
  else if (node.input().size() == 3)
  {
    std::string initializerWeightsName = node.input(1);
    std::string initializerBiasName = node.input(2);
    useBias = true;
    initializerWeights = Initializer(graph, initializerWeightsName);
    initializerBias = Initializer(graph, initializerBiasName);
    // layerParameters.push_back(arma::join_cols(
    //     arma::conv_to<arma::mat>::from(
    //         ConvertToColumnMajor(initializerWeights)),
    //     arma::conv_to<arma::mat>::from(
    //         ConvertToColumnMajor(initializerBias))));
    dag.Network()[layerIndex[0]]->Parameters() = arma::join_cols(
        arma::conv_to<arma::mat>::from(
            ConvertToColumnMajor(initializerWeights)),
        arma::conv_to<arma::mat>::from(
            ConvertToColumnMajor(initializerBias)));
  }
}

inline int FindConvMap(mlpack::DAGNetwork<>& dag,
                       onnx::GraphProto graph,
                       onnx::NodeProto node)
{
  // return ffn.Network().back()->OutputDimensions()[2];
  return 16;
}

} // namespace onnx_mlpack

#endif
