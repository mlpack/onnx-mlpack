#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "mlpack.hpp"
#include "onnx_pb.h"

using namespace std;

void AddSoftmax(mlpack::FFN<> &ffn, vector<arma::Mat<double>> &layerParameters);


#include "Softmax_impl.hpp"
#endif
