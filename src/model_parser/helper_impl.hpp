#include "helper.hpp"

string get::ModelInput(onnx::GraphProto &graph)
{
    vector<string> inputNames;
    vector<string> initializerNames;
    for (auto input : graph.input())
    {
        inputNames.push_back(input.name());
    }
    for (auto initializer : graph.initializer())
    {
        initializerNames.push_back(initializer.name());
    }

    // input for which no initializer will be the modelInput
    for (const auto &element : inputNames)
    {
        if (std::find(initializerNames.begin(), initializerNames.end(),
                      element) == initializerNames.end())
        {
            return element;
        }
    }
    // **** have to put error condition here
    cout << "error in finding the modelInput" << endl;
    return "";
}

vector<size_t> get::InputDimension(onnx::GraphProto &graph,
                                   const string &modelInput)
{
    vector<size_t> dimension;
    for (auto input : graph.input())
    {
        if (input.name() == modelInput)
        {
            int dim_size = input.type().tensor_type().shape().dim().size();
            for (int i = dim_size - 1; i > 0; i--)
            {
                dimension.push_back(input.type().tensor_type().shape().dim(i).dim_value());
            }
        }
    }
    return dimension;
}

// to get the indexs of all the nodes which will be tkaing the output of the current node
vector<int> get::DependentNodes(onnx::GraphProto &graph,
                                const string &nodeInput)
{
    vector<int> nodes;
    for (int i = 0; i < graph.node().size(); i++)
    {
        for (int j = 0; j < graph.node(i).input().size(); j++)
        {
            if (nodeInput == graph.node(i).input(j))
            {
                nodes.push_back(i);
            }
        }
    }
    return nodes;
}

vector<vector<int>> get::AdjacencyMatrix(onnx::GraphProto &graph)
{
    int totalNodes = graph.node().size();

    // creating the adjencList and inDegree
    vector<vector<int>> adj(totalNodes);
    for (int i = 0; i < totalNodes; i++)
    {
        for (string output : graph.node(i).output())
        {
            vector<int> nodeIndex = get::DependentNodes(graph,
                                                        output);
            for (int element : nodeIndex)
            {
                adj[i].push_back(element);
            }
        }
    }
    return adj;
}

vector<int> get::TopologicallySortedNodes(onnx::GraphProto &graph)
{
    int totalNodes = graph.node().size();

    // creating the adjencList and inDegree
    vector<vector<int>> adj = get::AdjacencyMatrix(graph);

    // ------ print the adj
    // int i = 0;
    // for (auto element : adj)
    // {
    //     cout << i << " => " << element << endl;
    //     i++;
    // }

    int visitedNode[totalNodes] = {0};
    stack<int> st;

    for (int i = 0; i < totalNodes; i++)
    {
        if (!visitedNode[i])
        {
            get::dfs(i, visitedNode, adj, st);
        }
    }
    vector<int> topologicalShort;
    while (!st.empty())
    {
        topologicalShort.push_back(st.top());
        st.pop();
    }
    return topologicalShort;
}

void get::dfs(int node, int visited[],
              const vector<vector<int>> &adj, stack<int> &st)
{
    visited[node] = 1;
    for (int neighbouringNode : adj[node])
    {
        if (!visited[neighbouringNode])
        {
            dfs(neighbouringNode, visited, adj, st);
        }
    }
    st.push(node);
}

const onnx::TensorProto &get::Initializer(onnx::GraphProto &graph,
                                          const string &initializerName)
{
    for (const onnx::TensorProto &init : graph.initializer())
    {
        if (initializerName == init.name())
        {
            return init;
        }
    }
    throw std::runtime_error("No initializer found with name " +
                             initializerName);
}

arma::fmat get::ConvertToColumnMajor(const onnx::TensorProto &initializer)
{
    // onnx initializer stores data in row major format
    // {N, C, H, W}
    vector<int> rowMajorDim(4, 1);
    int j = 3;
    int n_dims = initializer.dims().size();
    for (int i = n_dims - 1; i >= 0; i--)
    {
        rowMajorDim[j] = initializer.dims(i);
        j--;
    }

    int N = rowMajorDim[0]; // l
    int C = rowMajorDim[1]; // k
    int H = rowMajorDim[2]; // j
    int W = rowMajorDim[3]; // i
    vector<float> colMajorData;

    for (int l = 0; l < N; l++)
    {
        for (int k = 0; k < C; k++)
        {
            for (int j = 0; j < W; j++)
            {
                for (int i = 0; i < H; i++)
                {
                    // int colMajorIndex = l * (H * C * N) + k * (C * N) + j * (N) + i;
                    int rowMajorIndex = j + (i * W) + (k * W * H) + (l * C * W * H);
                    colMajorData.push_back(initializer.float_data(rowMajorIndex));
                }
            }
        }
    }

    arma::fmat matrix(colMajorData);
    return matrix;
}

vector<double> get::ConvertToRowMajor(const arma::mat &matrix,
                                      const vector<size_t> &outputDimension)
{
    int C = outputDimension[2];
    int H = outputDimension[1];
    int W = outputDimension[0];

    vector<double> returnValue;
    for (int i = 0; i < C; i++)
    {
        for (int j = 0; j < H; j++)
        {
            for (int k = 0; k < W; k++)
            {
                returnValue.push_back(matrix((j + (H * k) +
                                              (i * W * H)),
                                             0));
            }
        }
    }
    return returnValue;
}

void get::ImageToColumnMajor(arma::mat &matrix,
                             const vector<size_t> &outputDimension)
{
    int C = outputDimension[2];
    int H = outputDimension[1];
    int W = outputDimension[0];

    vector<double> returnValue;
    for (int i = 0; i < C; i++)
    {
        for (int j = 0; j < W; j++)
        {
            for (int k = 0; k < H; k++)
            {
                returnValue.push_back(matrix((j + (W * k) +
                                              (i * W * H)),
                                             0));
            }
        }
    }

    arma::mat newMat(returnValue);
    matrix = newMat;
    // return returnValue;
}

void get::DrawRectangle(const string &imagePath, const string &finalImagePath,
                        int r1, int c1, int r2, int c2, const vector<int> &imageDimension)
{
    // Extracting image, Input
    int W = imageDimension[0];
    int H = imageDimension[1];
    int C = imageDimension[2];
    mlpack::data::ImageInfo imageInfo(W, H, C, 1);
    string fileName = imagePath;
    arma::Mat<double> imageMat;
    mlpack::data::Load<double>(fileName, imageMat, imageInfo, false);
    // ImageMatrx => rgb rgb => along column
    // we want int => rrr...ggg...bbb...

    // r1
    for (int i = c1; i < c2; i++)
    {
        imageMat(0 + (C * i) + (C * W * r1), 0) = 255;
        imageMat(1 + (C * i) + (C * W * r1), 0) = 0;
        imageMat(2 + (C * i) + (C * W * r1), 0) = 0;
    }
    // c1
    for (int i = r1; i < r2; i++)
    {
        imageMat(0 + (C * c1) + (C * W * i), 0) = 255;
        imageMat(1 + (C * c1) + (C * W * i), 0) = 0;
        imageMat(2 + (C * c1) + (C * W * i), 0) = 0;
    }
    // r2
    for (int i = c1; i < c2; i++)
    {
        imageMat(0 + (C * i) + (C * W * r2), 0) = 255;
        imageMat(1 + (C * i) + (C * W * r2), 0) = 0;
        imageMat(2 + (C * i) + (C * W * r2), 0) = 0;
    }
    // c2
    for (int i = r1; i < r2; i++)
    {
        imageMat(0 + (C * c2) + (C * W * i), 0) = 255;
        imageMat(1 + (C * c2) + (C * W * i), 0) = 0;
        imageMat(2 + (C * c2) + (C * W * i), 0) = 0;
    }

    mlpack::data::Save(finalImagePath, imageMat, imageInfo, true);
}

void get::DrawRectangleOnCsv(arma::mat &matrix, int r1, int c1,
                             int r2, int c2, const vector<int> &imageDimension)
{
    // Extracting image, Input
    int W = imageDimension[0];
    int H = imageDimension[1];
    int C = imageDimension[2];
    if (r1 < 0)
        r1 = 0;
    if (r2 >= H)
        r2 = H - 1;
    if (c1 < 0)
        c1 = 0;
    if (c2 >= W)
        c2 = W - 1;

    // r1
    for (int i = c1; i < c2; i++)
    {
        matrix((0 * W * H) + (H * i) + (r1), 0) = 255;
        matrix((1 * W * H) + (H * i) + (r1), 0) = 0;
        matrix((2 * W * H) + (H * i) + (r1), 0) = 0;
    }
    // c1
    for (int i = r1; i < r2; i++)
    {
        matrix((0 * W * H) + (H * c1) + (i), 0) = 255;
        matrix((1 * W * H) + (H * c1) + (i), 0) = 0;
        matrix((2 * W * H) + (H * c1) + (i), 0) = 0;
    }
    // r2
    for (int i = c1; i < c2; i++)
    {
        matrix((0 * W * H) + (H * i) + (r2), 0) = 255;
        matrix((1 * W * H) + (H * i) + (r2), 0) = 0;
        matrix((2 * W * H) + (H * i) + (r2), 0) = 0;
    }
    // c2
    for (int i = r1; i < r2; i++)
    {
        matrix((0 * W * H) + (H * c2) + (i), 0) = 255;
        matrix((1 * W * H) + (H * c2) + (i), 0) = 0;
        matrix((2 * W * H) + (H * c2) + (i), 0) = 0;
    }
}
