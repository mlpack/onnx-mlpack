#include "Softmax.hpp"

vector<size_t> AddSoftmax(mlpack::DAGNetwork<> &dag){
    // layerParameters.push_back(arma::Mat<double>());
    size_t a = dag.Add<mlpack::Softmax>();

    
    cout<<"Added mlpack::Softmax Layer"<<endl;
    return {a};
}