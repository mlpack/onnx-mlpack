#include "converter.hpp"
#include <cmath>

vector<string> class_names = {"Aeroplane", "Bicycle", "Bird", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cow",
                              "Dining Table", "Dog", "Horse", "Motorbike", "Person", "Potted Plant", "Sheep", "Sofa",
                              "Train", "TV"};

vector<float> anchors = {1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52};

double sigmoid(double x)
{
    return 1.0 / (1 + exp(-x));
}

vector<double> softmax(vector<double> v)
{
    auto max_it = max_element(v.begin(), v.end());
    double max = *max_it;
    double sum = 0;
    for (int i = 0; i < v.size(); i++)
    {
        v[i] = v[i] - max;
        v[i] = exp(v[i]);
        sum += v[i];
    }

    for (int i = 0; i < v.size(); i++)
    {
        v[i] = v[i] / sum;
    }

    return v;
}

// v1 = x1, y1, x2, y2
// v2 = x3, y3, x4, y4
float IOU(const std::vector<int> &v1, const std::vector<int> &v2)
{
    int interior_x1 = std::max(v1[0], v2[0]);
    int interior_y1 = std::max(v1[1], v2[1]);
    int interior_x2 = std::min(v1[2], v2[2]);
    int interior_y2 = std::min(v1[3], v2[3]);

    int intersection_width = interior_x2 - interior_x1;
    int intersection_height = interior_y2 - interior_y1;

    // Check if the rectangles actually overlap
    if (intersection_width <= 0 || intersection_height <= 0)
        return 0.0f;

    int intersection_area = intersection_width * intersection_height;

    int area1 = (v1[2] - v1[0]) * (v1[3] - v1[1]);
    int area2 = (v2[2] - v2[0]) * (v2[3] - v2[1]);

    int union_area = area1 + area2 - intersection_area;

    return static_cast<float>(intersection_area) / static_cast<float>(union_area);
}

vector<pair<int, pair<float, vector<int>>>> NonMaxSupression(vector<vector<int>> &boxes,
                                                             vector<float> &confidences,
                                                             vector<int> &ids,
                                                             float confidenceThresold,
                                                             float iouThresold)
{
    // removing all the elements with confidence less that the thresold
    for (int i = 0; i < confidences.size(); i++)
    {
        if (confidences[i] < confidenceThresold)
            ids[i] = -1;
    }

    // seprating the same id element seprately
    map<int, vector<pair<float, vector<int>>>> all;
    for (int i = 0; i < ids.size(); i++)
    {
        if (ids[i] == -1)
            continue;
        pair<float, vector<int>> a({0, vector<int>(4)});
        a.first = confidences[i];
        a.second = boxes[i];
        all[ids[i]].push_back(a);
    }

    vector<pair<int, pair<float, vector<int>>>> final;
    for (auto itr : all)
    {

        int id = itr.first;
        float maxConfidence = INT_MIN;
        vector<int> maxCBox(4);
        for (auto element : itr.second)
        {
            if (element.first > maxConfidence)
            {
                maxConfidence = element.first;
                maxCBox = element.second;
            }
        }

        for (auto element : itr.second)
        {
            if (IOU(element.second, maxCBox) < iouThresold || IOU(element.second, maxCBox) == 1)
            {
                pair<int, pair<float, vector<int>>> p;
                p.first = id;
                p.second.first = element.first;
                p.second.second = element.second;
                final.push_back(p);
            }
        }
    }
    return final;
}

int main()
{
    std::cout << std::fixed << std::setprecision(10);

    // generating the onnx graph
    string onnxFilePath = "tinyyolo-v2.3-o8.onnx";
    onnx::GraphProto graph = getGraph(onnxFilePath);

    // getting the model from the graph
    mlpack::FFN<> generatedModel = converter(graph);

    int H = 416;
    int W = 416;
    int C = 3;
    for (int i = 1; i < 11; i++)
    {
        // mlpack::image load is little inaccurate so we are first converting the
        // image into csv format using opencv externally and passing the image to the model

        // loading image from the csv file
        string loat_path = "/home/kumarutkarsh/Desktop/onnx-mlpack/example/yolo-tiny/csv_images/" + to_string(i) + ".csv";
        arma::mat imageMatrix;
        bool load_status = imageMatrix.load(loat_path, arma::csv_ascii);
        get::ImageToColumnMajor(imageMatrix, {416, 416, 3});

        //**  however image can be loaded from mlpack::imageLoad and used the data upto certain accuracy
        mlpack::data::ImageInfo imageInfo(416, 416, 3, 1);
        // string fileName = "image(416-416)/" + to_string(i) + ".jpg";
        // arma::Mat<double> imageMat;
        // mlpack::data::Load<double>(fileName, imageMat, imageInfo, false);
        // // ImageMatrx => rgb rgb
        // //  we want int => rrr...ggg...bbb...
        // vector<double> imageVector(H * W * C, 0);
        // for (int i = 0; i < C; i++)
        // {
        //     for (int j = 0; j < W; j++)
        //     {
        //         for (int k = 0; k < H; k++)
        //         {
        //             imageVector[k + (j * H) + (i * H * W)] = imageMat(i + (C * W * k) + (C * j), 0);
        //         }
        //     }
        // }
        // arma::mat imageMatrix(imageVector);
        //--------------------------------------------------------

        // getting the output from the prediction method
        arma::mat output;
        generatedModel.Predict(imageMatrix, output);
        arma::cube finalOutput(output.memptr(), 13, 13, 125, false, true);
        // finalOutput.print("final output");

        // get the most confident object
        int numClasses = 20;
        vector<vector<int>> boxes;
        vector<float> confidences;
        vector<int> ids;
        for (int cy = 0; cy < 13; cy++)
        {
            for (int cx = 0; cx < 13; cx++)
            {
                map<double, vector<double>> confidence_probablity;
                for (int c = 0; c < 5; c++)
                {
                    int channel = c * (numClasses + 5);

                    double tx = finalOutput(cx, cy, channel + 0);
                    double ty = finalOutput(cx, cy, channel + 1);
                    double tw = finalOutput(cx, cy, channel + 2);
                    double th = finalOutput(cx, cy, channel + 3);
                    double tc = finalOutput(cx, cy, channel + 4);
                    vector<double> probs;
                    for (int i = 0; i < 20; i++)
                        probs.push_back(finalOutput(cx, cy, channel + 5 + i));

                    int x = (cx + sigmoid(tx)) * 32;
                    int y = (cy + sigmoid(ty)) * 32;
                    int h = exp(tw) * 32 * anchors[2 * c];
                    int w = exp(th) * 32 * anchors[2 * c + 1];

                    int x1 = x - (w / 2);
                    int y1 = y - (h / 2);
                    int x2 = x + (w / 2);
                    int y2 = y + (h / 2);
                    vector<int> box = {x1, y1, x2, y2};

                    double confidence = sigmoid(tc);

                    probs = softmax(probs);
                    auto itr = max_element(probs.begin(), probs.end());
                    int bestClassIds = std::distance(probs.begin(), itr);
                    float bestClassProb = probs[bestClassIds];

                    float confidenceInClass = confidence * bestClassProb;

                    boxes.push_back(box);
                    confidences.push_back(confidenceInClass);
                    ids.push_back(bestClassIds);
                }
            }
        }

        // Removing all the least probable boxed and overlaping boxed
        // <id, <confidence score, bonding box>>
        vector<pair<int, pair<float, vector<int>>>> result =
            NonMaxSupression(boxes, confidences, ids, 0.4, 0.4);

        cout<<"for "<<i<<".png"<<endl;
        for (auto output : result)
        {
            int r1 = output.second.second[0];
            int c1 = output.second.second[1];
            int r2 = output.second.second[2];
            int c2 = output.second.second[3];
            // modify the imageMatrix (draw rectangle)
            get::DrawRectangleOnCsv(imageMatrix, r1, c1, r2, c2, {416, 416, 3});
            cout<<"Detedted-Object "<<class_names[output.first]<<"; Confidence-Score "<<output.second.first<<endl;
        }
        cout<<endl;

        // transforming the image so that it can be saved correctly by mlpack::save
        vector<double> finalImage;
        for (int i = 0; i < 416; i++)
        {
            for (int j = 0; j < 416; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    finalImage.push_back(imageMatrix(i + (H * W * k) + (H * j), 0));
                }
            }
        }
        arma::mat f(finalImage);
        string save_path = "yolo_output_image/" + to_string(i) + ".png";
        mlpack::data::Save(save_path, f, imageInfo, true);
    }

    return 0;
}



// // *** below code is for debugging purpose
// int mul(vector<size_t> v)
// {
//     int ans = 1;
//     for (int element : v)
//     {
//         ans *= element;
//     }
//     return ans;
// }

/*
// Extracting image, Input
mlpack::data::ImageInfo imageInfo(416, 416, 3, 1);
string fileName = "resized_images/10.png";
arma::Mat<double> imageMat;
mlpack::data::Load<double>(fileName, imageMat, imageInfo, false);
//ImageMatrx => rgb rgb
// we want int => rrr...ggg...bbb...
int H = 416;
int W = 416;
int C = 3;
vector<double> imageVector(H*W*C, 0);
for(int i=0; i<C; i++){
    for(int j=0; j<W; j++){
        for(int k=0; k<H; k++){
            imageVector[k + (j*H) + (i*H*W)] = imageMat(i + (C*W*k) + (C*j), 0);
        }
    }
}
arma::mat imageMatrix_(imageVector);
vector<double> v = ConvertToRowMajor(imageMatrix_, {W, H, C});
// cout<<"--->"<<imageVector<<endl;
for(int i=0; i<10; i++){
    cout<<imageVector[i]<<endl;
}
*/

// forward pass one by one
/*
arma::Mat<double> input = imageMatrix;
int i=1;
for (auto layer : generatedModel.Network())
{
    arma::Mat<double> output(mul(layer->OutputDimensions()), 1, arma::fill::ones);
    layer->Forward(input, output);
    input = output;

    // arma::mat A = input.submat(0, 0, 5, 0);
    vector<double> v = ConvertToRowMajor(input, layer->OutputDimensions());


    // printing the output dimension
    // Set precision to 10 decimal places
    std::cout << std::fixed << std::setprecision(10);

    // Use raw_print to have more control over formatting
    // A.raw_print(std::cout);
    // cout << " output dimensions " << i << " " << output.n_rows << endl;
    cout<<"output Dimension "<<i<<layer->OutputDimensions()<<endl;
    // A.raw_print(std::cout);
    for(int i=0; i<5; i++){
        cout<<v[i]<<" ";
    }
    cout<<endl<<endl;
    // cout<<A<<endl<<endl;
    i++;
}
*/