#ifndef RESHAPE_HPP
#define RESHAPE_HPP

#include "mlpack.hpp"
#include <onnx/onnx_pb.h>
#include "../onnx_mlpack/utils.hpp"
#include "../onnx_mlpack/helper.hpp"

using namespace std;

inline vector<size_t> AddReshape(mlpack::DAGNetwork<> &dag, onnx::GraphProto graph,
                onnx::NodeProto node, map<string, double> onnxOperatorAttribute);

class Reshape : public mlpack::Identity<arma::mat>
{
public:
    vector<size_t> modifiedDimension; // w, h, c
    inline Reshape(vector<size_t> modifiedDimension) : modifiedDimension(modifiedDimension) {}

    inline void ComputeOutputDimensions()
    {
        outputDimensions = modifiedDimension;
    }
};

inline vector<int> FindReshapedDimension(onnx::GraphProto graph, onnx::NodeProto node);

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

#include "Reshape_impl.hpp"
#endif
