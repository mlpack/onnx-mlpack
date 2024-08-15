#include "Relu.hpp"

void AddRelu(mlpack::FFN<> &ffn, vector<arma::Mat<double>> &layerParameters){
    layerParameters.push_back(arma::Mat<double>());
    ffn.Add(new mlpack::LeakyReLU(0));
    cout<<"Added the Relu layer"<<endl;
}