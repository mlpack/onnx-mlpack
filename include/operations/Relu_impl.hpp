#include "Relu.hpp"

inline vector<size_t> AddRelu(mlpack::DAGNetwork<> &dag){
    // layerParameters.push_back(arma::Mat<double>());
    size_t a = dag.Add<mlpack::LeakyReLU>(0);
    
    cout<<"Added mlpack::Relu Layer"<<endl;
    return {a};
}
