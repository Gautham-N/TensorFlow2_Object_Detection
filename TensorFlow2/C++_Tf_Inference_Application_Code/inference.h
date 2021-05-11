#ifndef INFERENCE_H
#define INFERENCE_H
#include <string>
#include "tensorflow/c/c_api.h"
#include <iostream>
#include <opencv2/core/types_c.h>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
using namespace cv;
using namespace std;

struct Results {
    float* boxes;
    float* scores;
    float* label_ids;
};

class Inference
{
private:
    TF_Graph* graph;
    TF_Buffer* graph_def;
    TF_ImportGraphDefOptions* graph_opts;
    TF_Status* graph_status;
    TF_SessionOptions* sess_opts = TF_NewSessionOptions();
    TF_Status* sess_status;
    TF_Session* sess;
    TF_Operation* input_op;
    TF_Output input_opout;
    std::vector<TF_Output> input_ops;
    std::vector<TF_Tensor*> input_values;
    TF_Operation* boxes_op;
    TF_Operation* scores_op;
    TF_Operation* classes_op;
    TF_Operation* num_detections_op;
    TF_Output boxes_opout, scores_opout, classes_opout, num_detections_opout;
    std::vector<TF_Output> output_ops;
    std::vector<TF_Tensor*> output_values;
public:

    Inference();

    //-----------Loading-the-Model----------------//

    bool LoadModel(string Path);

    //------------Predict-Image--------------------//

    Results Predict(string ImagePath);


};

#endif // INFERENCE_H
