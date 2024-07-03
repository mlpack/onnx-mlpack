# Before building from source uninstall any existing versions of onnx 
pip uninstall onnx

# install protobuf
sudo apt-get install python3-pip python3-dev libprotobuf-dev protobuf-compiler

cd build_onnx
unzip onnx_main.zip
cd onnx-main

# installing the shared lib
mkdir build
cd build
cmake DONNX_USE_PROTOBUF_SHARED_LIBS=ON ..
sudo make install
