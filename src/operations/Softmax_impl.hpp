#include "Softmax.hpp"

void AddSoftmax(mlpack::FFN<> &ffn, vector<arma::Mat<double>> &layerParameters){
    layerParameters.push_back(arma::Mat<double>());
    ffn.Add(new mlpack::Softmax());
    cout<<"Added the Softmax layer"<<endl;
}