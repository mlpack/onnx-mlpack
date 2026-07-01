# Use this Makefile as a starting point to adapt to your system.
# It is kept intentionally as simple as possible for demonstration purposes.

all: convert_mobilenet.cpp
	g++ -std=c++17 -DONNX_ML=1 -O3 -o convert_mobilenet convert_mobilenet.cpp -larmadillo -lonnx -lprotobuf
