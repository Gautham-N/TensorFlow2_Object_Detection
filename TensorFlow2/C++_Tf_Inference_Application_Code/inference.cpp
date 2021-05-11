#include "inference.h"

Inference::Inference()
{

}

void Deallocator(void* data, size_t length, void* arg)
{

}

//----------------------------Loading-the-Model---------------------------------------//

bool Inference::LoadModel(string Path)
{
    try
    {
        graph = TF_NewGraph();
        graph_status = TF_NewStatus();

        TF_Buffer* RunOpts = NULL;

        const char* saved_model_dir =Path.c_str(); // Path of the model
        const char* tags = "serve";
        int ntags = 1;

        sess = TF_LoadSessionFromSavedModel(sess_opts, RunOpts, saved_model_dir, &tags, ntags, graph, NULL, graph_status);
        if(TF_GetCode(graph_status) == TF_OK)
        {
            cout<<("TF_LoadSessionFromSavedModel OK\n")<<endl;
            return true;
        }
        else
        {
            cout<<(TF_Message(graph_status))<<endl;
            return false;
        }

    }
    catch (...)
    {
        cout<<"Error in Loading Model"<<endl;
        return false;
    }
}

//----------------------------Predict-Image---------------------------------------//

Results Inference::Predict(string ImagePath)
{
    try
    {
        std::vector<TF_Output> 	input_tensors, output_tensors;
        std::vector<TF_Tensor*> input_values, output_values;
        Mat image=imread(ImagePath);
        int num_dims = 4;
        std::int64_t input_dims[4] = {1, image.rows, image.cols, 3};
        int num_bytes_in = image.cols * image.rows * 3;

        input_tensors.push_back({TF_GraphOperationByName(graph, "serving_default_input_tensor"),0});
        input_values.push_back(TF_NewTensor(TF_UINT8, input_dims, num_dims, image.data, num_bytes_in, &Deallocator, 0));


        output_tensors.push_back({ TF_GraphOperationByName(graph, "StatefulPartitionedCall"),1});
        output_values.push_back(nullptr);

        output_tensors.push_back({ TF_GraphOperationByName(graph, "StatefulPartitionedCall"),2 });
        output_values.push_back(nullptr);

        output_tensors.push_back({ TF_GraphOperationByName(graph, "StatefulPartitionedCall"),4 });
        output_values.push_back(nullptr);

        TF_Status* status = TF_NewStatus();
        TF_SessionRun(sess, nullptr,
                      &input_tensors[0], &input_values[0], input_values.size(),
                &output_tensors[0], &output_values[0], 3, //3 is the number of outputs count..
                nullptr, 0, nullptr, status
                );	if (TF_GetCode(status) != TF_OK)
        {
            cout<<"ERROR: SessionRun"<<TF_Message(status)<<endl;
        }

        Results PredictedResults;
        PredictedResults.label_ids = static_cast<float_t*>(TF_TensorData(output_values[1]));
        PredictedResults.scores = static_cast<float_t*>(TF_TensorData(output_values[2]));
        PredictedResults.boxes = static_cast<float_t*>(TF_TensorData(output_values[0]));

        return PredictedResults;
    }
    catch (...)
    {
        cout<<"Error in Predicting the Image"<<endl;
        return {};
    }
}
