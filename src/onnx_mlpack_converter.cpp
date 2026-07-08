/**
 * @file onnx_mlpack_converter.cpp
 * @author Ryan Curtin
 *
 * This program converts an ONNX network to an equivalent mlpack DAGNetwork and
 * serializes it to file.
 *
 * The ONNX/mlpack converter is free software; you may redistribute it and/or
 * modify it under the terms of the 3-clause BSD license.  You should have
 * received a copy of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>
#include <onnx_mlpack.hpp>

using namespace mlpack;
using namespace std;

int main(int argc, char** argv)
{
  // TODO: better input handling
  if (argc != 3)
  {
    cerr << "Usage: " << argv[0] << " input_network.onnx output_network.bin"
        << endl;
    exit(1);
  }

  const string inputNetwork(argv[1]);
  const string outputNetwork(argv[2]);

  DAGNetwork<> dag = onnx_mlpack::Convert(inputNetwork);

  if (dag.Network().size() == 0)
    throw std::runtime_error("Converted DAG is empty!  Aborting.");

  Save(outputNetwork, dag, Fatal);
}
