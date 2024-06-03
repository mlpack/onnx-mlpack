#include "softmax.hpp"

void AddSoftmax(mlpack::FFN<> &ffn){
    ffn.Add(new mlpack::Softmax());
    cout<<"added the softmax"<<endl;
}