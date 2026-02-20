#ifndef GLOBALAVERAGEPOOL_HPP
#define GLOBALAVERAGEPOOL_HPP

#include "mlpack.hpp"
#include <onnx/onnx_pb.h>

using namespace std;

inline vector<size_t> AddGlobalAveragePool(mlpack::DAGNetwork<> &dag);


#include "GlobalAveragePool_impl.hpp"
#endif
