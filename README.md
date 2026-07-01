
# ONNX-mlpack Translator

<div align="center">
  <a href="http://mlpack.org">
    <img src="img/onnx-mlpack.png" alt="ONNX-mlpack Translator" style="max-width: 45%; height: auto;">
  </a>
  <br>
</div>

The ONNX-mlpack converter can take any ONNX graph and convert it to an mlpack
neural network.  This is done by matching individual mlpack layers to subgraphs
of the ONNX network using a rule matching engine, and then selecting the overall
best match of the network.

The conversion can be done either with a standalone command-line program, or
directly in C++.

## Setup

### 1. Dependencies

Make sure that you have, available on your system,

 - C++17 compiler
 - CMake
 - [mlpack](https://www.mlpack.org) and its dependencies
   [ensmallen](https://www.ensmallen.org),
   [cereal](https://github.com/USCILab/cereal), and
   [Armadillo](https://arma.sourceforge.net); if not installed, these will be
   autodownloaded during the CMake configuration
 - ONNX (on Debian, `libonnx-dev` is sufficient)
 - Protobuf (on Debian, `libprotobuf-dev` is sufficient)

On Debian, all these dependencies can be installed with:

```sh
sudo apt-get install cmake make g++ libmlpack-dev libonnx-dev libprotobuf-dev
```

On Fedora/RHEL, this command can be used:

```sh
sudo dnf install cmake gcc-c++ mlpack-devel onnx-devel protobuf-devel
```

### 2. Configuration

Create a build directory and use CMake to configure:

```sh
mkdir build/
cd build/
cmake ../
```

Note that this will automatically fetch mlpack and its dependencies if they are
not found on the system.

### 3. Build the converter

Now build the converter in the build directory.

```sh
make onnx_mlpack_converter
```

### 4. Install the converter

Optionally, install the converter and header files to the system:

```sh
sudo make install
```

## Usage

The command-line converter is very simple to use:

```sh
./onnx_mlpack_converter <input_network.onnx> <output_mlpack_network.bin>
```

Once the converter has been run, you can load the network into mlpack as a
`DAGNetwork` in a C++ program:

```c++
#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>

using namespace mlpack;

int main()
{
  DAGNetwork<> network;
  Load("output_mlpack_network.bin", network);

  // now you can use network.Predict(), network.Train(), etc.
}
```

## Converting in C++

Instead of calling `onnx_mlpack_converter`, you can also just convert directly
in C++:

```c++
#include <mlpack.hpp>
#include <onnx_mlpack.hpp>

int main()
{
  mlpack::DAGNetwork<> result = onnx_mlpack::Convert("input_model.onnx");

  // or load the ONNX graph manually, then simplify and convert:

  onnx::GraphProto graph = onnx_mlpack::Load("input_model.onnx");
  onnx_mlpack::Simplify(graph);
  mlpack::DAGNetwork<> result2 = onnx_mlpack::Convert(graph);
}
```

Compiling in C++ requires a minimum of C++17, specifying an ONNX macro, and
linking against Armadillo, ONNX, and Protobuf:

```sh
g++ -std=c++17 -DONNX_ML=1 -o program program.cpp -larmadillo -lonnx -lprotobuf
```

See the [simple example](examples/convert_tinyyolo.cpp) for a working example
and `Makefile`.

## C++ API Documentation

### `Load()`

 - `onnx::GraphProto graph = onnx_mlpack::Load(filename)`
   * Load the ONNX graph from `filename` (a `std::string`) and perform shape
     inference for all tensors in the graph.

### `Simplify()`

 - `onnx_mlpack::Simplify(graph)`
   * Given `graph`, a loaded ONNX graph of type `onnx::GraphProto&`, simplify
     the graph in-place:
     - `Identity` operators are removed.
     - Unnecessary `Reshape` operators are removed or inlined into the tensors
       they are applied to.
     - `Add` and `Mul` operators that have no effect are removed.

   * This modifies `graph` in-place and does not return a new graph.

   * If you have loaded an ONNX graph manually or with `Load()`, this function
     *must* be called before calling `Convert(graph)`.

### `Convert()`

 - `mlpack::DAGNetwork<> result = onnx_mlpack::Convert(filename)`
   * Load the ONNX graph from `filename` (a `std::string`), simplify the graph,
     and convert to an mlpack `DAGNetwork`.

   * This handles all graph simplifications and preprocessing automatically.

   * If the graph cannot be fully matched to an mlpack `DAGNetwork`, a
     `std::runtime_error` will be thrown with more details.

 - `mlpack::DAGNetwork<> result = onnx_mlpack::Convert(graph)`
   * Given `graph`, a loaded ONNX graph of type `onnx::GraphProto`, convert to
     an `mlpack::DAGNetwork<>`.

   * Make sure `Simplify()` has been called on the graph first!

   * If `Load()` was not used to load the graph, make sure also that shape
     inference has been performed with
     [onnx::shape_inference::InferShapes()](https://github.com/onnx/onnx/blob/main/docs/ShapeInference.md)
     or equivalent.

   * If the graph cannot be fully matched to an mlpack `DAGNetwork`, a
     `std::runtime_error` will be thrown with more details.

## Notes

This repository is under active development!  At this particular moment, support
is not available for all ONNX operators, but it is actively being expanded.  If
you have a network that does not match properly, or encounter other problems,
please feel free to open a bug report and we will look into the issue.
