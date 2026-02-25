/**
 * @file reshape.hpp
 * @author Kumar Utkarsh
 *
 * Handling of ONNX Reshape operator conversion.
 */
#ifndef ONNX_MLPACK_OPERATIONS_RESHAPE_HPP
#define ONNX_MLPACK_OPERATIONS_RESHAPE_HPP

#include "../utils.hpp"
#include "../helper.hpp"

namespace onnx_mlpack {

inline std::vector<size_t> AddReshape(
    mlpack::DAGNetwork<> &dag,
    onnx::GraphProto graph,
    onnx::NodeProto node,
    std::map<std::string, double> onnxOperatorAttribute);

class Reshape : public mlpack::Identity<arma::mat>
{
 public:
  std::vector<size_t> modifiedDimension; // w, h, c

  inline Reshape(std::vector<size_t> modifiedDimension) :
      modifiedDimension(modifiedDimension) { }

  inline void ComputeOutputDimensions()
  {
    outputDimensions = modifiedDimension;
  }
};

inline std::vector<int> FindReshapedDimension(
    onnx::GraphProto graph,
    onnx::NodeProto node);

//     void Forward(
//         const arma::mat &input, arma::mat &output)
//     {
//         size_t n_batches = input.n_cols;
//         // make each col into cube
//         // then apply the reshape
//         // again convert the reshaped one into col vector
//         // add the col_vector in the output
//         for(int i=0; i<n_batches; i++){
//             arma::colvec colVec = input.col(i);
//             arma::cube Cube = arma::reshape(colVec, inputDimensions[1], inputDimensions[0], inputDimensions[2]);
//             Cube.reshape(outputDimension[1], outputDimension[0], outputDimension[2]);
//             arma::colvec colVec_ = arma::vectorise(cube);
//             output.col(i) = colVec_
//         }

//         output = input * scalar;
//     }

} // namespace onnx_mlpack

#include "reshape_impl.hpp"

#endif
