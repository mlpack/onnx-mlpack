#include "Softmax.hpp"

void AddSoftmax(mlpack::FFN<> &ffn){
    ffn.Add(new mlpack::Softmax());
    cout<<"Added the Softmax layer"<<endl;
}