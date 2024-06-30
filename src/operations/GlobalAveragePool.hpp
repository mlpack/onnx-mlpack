#ifndef GLOBALAVERAGEPOOL_HPP
#define GLOBALAVERAGEPOOL_HPP

#include "mlpack.hpp"
#include "onnx_pb.h"

using namespace std;

void AddGlobalAveragePool(mlpack::FFN<> &ffn, vector<arma::Mat<double>> &layerParameters);


#include "GlobalAveragePool_impl.hpp"
#endif
