#include "Relu.hpp"

void AddRelu(mlpack::FFN<> &ffn){
    ffn.Add(new mlpack::LeakyReLU());
    cout<<"Added the Relu layer"<<endl;
}