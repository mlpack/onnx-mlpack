#ifndef GLOBALAVERAGEPOOL_HPP
#define GLOBALAVERAGEPOOL_HPP

#include "mlpack.hpp"
#include "onnx_pb.h"

using namespace std;

vector<size_t> AddGlobalAveragePool(mlpack::DAGNetwork<> &dag);


#include "GlobalAveragePool_impl.hpp"
#endif
