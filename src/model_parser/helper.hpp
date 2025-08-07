#ifndef HELPER_HPP
#define HELPER_HPP

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <stack>
#include "onnx_pb.h"

using namespace std;

namespace get
{
    /**
     * @brief Retrieve the name of the input node from an ONNX graph.
     * This input name is crucial as it allows us to fetch the dimensions
     * of the input features. For example, in a YOLO model, the input name
     * might be "image1", which will later be used in `InputDimension`
     * to obtain the dimensions of the input feature.
     * 
     * @param graph The ONNX graph.
     * @return The name of the model input.
     */
    string ModelInput(onnx::GraphProto& graph);

    /**
     * @brief Get the dimensions of the input features using the name obtained from `ModelInput`.
     * 
     * @param graph The ONNX graph.
     * @param modelInput The name of the model input obtained from `ModelInput`.
     * @return A vector containing the dimensions {channels, height, width}.
     */
    vector<size_t> InputDimension(onnx::GraphProto& graph, const string& modelInput);

    // DependentNodes, AdjacencyMatrix, TopologicallySortedNodes, dfs
    // these methods are made to get the node in topological order
    vector<int> DependentNodes(onnx::GraphProto& graph, const string& nodeInput);
    vector<vector<int>> AdjacencyMatrix(onnx::GraphProto& graph);
    vector<int> TopologicallySortedNodes(onnx::GraphProto& graph, vector<vector<int>> &adj);
    void dfs(int node, int visited[], const vector<vector<int>>& adj, stack<int>& st);

    /**
     * @brief Retrieve the initializer from the ONNX graph by its name.
     * 
     * @param graph The ONNX graph.
     * @param initializerName The name of the initializer.
     * @return A reference to the ONNX TensorProto representing the initializer.
     */
    const onnx::TensorProto& Initializer(onnx::GraphProto& graph, const string& initializerName);

    /**
     * @brief Convert ONNX initializer data from row-major to column-major format.
     * 
     * ONNX stores weight data in row-major format, but mlpack expects the data in column-major format.
     * This function handles the conversion.
     * 
     * @param initializer Reference to the ONNX initializer.
     * @return The converted matrix in column-major format.
     */
    arma::fmat ConvertToColumnMajor(const onnx::TensorProto& initializer);

    /**
     * @brief Convert an mlpack column-major matrix into row-major format.
     * 
     * This utility function is useful for comparing mlpack layer outputs with ONNX runtime outputs
     * to ensure successful weight transfer.
     * 
     * @param matrix The mlpack matrix in column-major format.
     * @param outputDimension The desired output dimensions {width, height, channels}.
     * @return A vector containing the data in row-major format.
     */
    vector<double> ConvertToRowMajor(const arma::mat& matrix, const vector<size_t>& outputDimension);

    /**
     * @brief Convert a loaded image into column-major format.
     * 
     * mlpack’s image loading does not inherently provide row-major or column-major formats,
     * so this function is necessary to convert the loaded image into the correct column-major
     * format for mlpack models.
     * 
     * @param matrix The matrix containing image data.
     * @param outputDimension The dimensions of the loaded image {width, height, channels}.
     */
    void ImageToColumnMajor(arma::mat& matrix, const vector<size_t>& outputDimension);

    /**
     * @brief Load an image, draw a rectangle on it, and save the modified image.
     * 
     * Uses mlpack’s image load method to load the image, draws a rectangle with
     * a two-pixel width, and saves the modified image.
     * 
     * @param imagePath The path to the image to be edited.
     * @param finalImagePath The path to save the modified image.
     * @param r1, c1, r2, c2 Coordinates of the diagonally opposite vertices of the rectangle.
     * @param imageDimension The dimensions of the image {width, height, channels}.
     */
    void DrawRectangle(const string& imagePath, const string& finalImagePath, int r1, int c1, int r2, int c2, const vector<int>& imageDimension);

    /**
     * @brief Draw a rectangle on an image matrix.
     * 
     * This function modifies the image matrix by drawing a rectangle on it.
     * 
     * @param matrix The matrix containing the image pixel values in column-major format.
     * @param r1, c1, r2, c2 Coordinates of the diagonally opposite vertices of the rectangle.
     * @param imageDimension The dimensions of the image {width, height, channels}.
     */
    void DrawRectangleOnCsv(arma::mat& matrix, int r1, int c1, int r2, int c2, const vector<int>& imageDimension);
} // namespace get

#include "helper_impl.hpp"
#endif // HELPER_HPP
