/**
 * @file onnx_mlpack_converter.cpp
 * @author Ryan Curtin
 *
 * This program converts an ONNX network to an equivalent mlpack DAGNetwork and
 * serializes it to file.
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

  onnx::GraphProto graph = onnx_mlpack::GetGraph(inputNetwork);
  DAGNetwork<> dag = onnx_mlpack::Convert(graph);

  if (dag.Network().size() == 0)
    throw std::runtime_error("Converted DAG is empty!  Aborting.");

  Save(outputNetwork, dag, Fatal);
}
