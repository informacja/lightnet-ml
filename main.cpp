#include <iostream>

#include "src/nnclassifier.h"
#include "src/dataset.h"

using namespace std;
using namespace LightNet;

void trainAndSave();
void loadAndPredict();

int main()
{
    trainAndSave();
    return 0;
}

void trainAndSave(){

    Dataset dataset("./../lightnet-ml/data/emg_features1.csv", true);
    dataset.scale();

    Dataset testData = dataset.splitTestData(10);

    NNClassifier net({dataset.getInputCount(), 10, dataset.getUniqueTargetCount()}, dataset);
    net.train(10);

    std::vector<NNClassifier::Prediction> predictions = net.predict(testData);

    for(NNClassifier::Prediction prediction : predictions){
        cout << "Predicted: " << prediction.predictedEncodedTarget << " Actual: " << prediction.actualEncodedTarget << " Conf: " << prediction.confidence << endl;
    }

    if(net.save("./../lightnet-ml/data/model.json")){
        std::cout << "saved!" << std::endl;
    }else {
        std::cout << "failed to save!" << std::endl;
    }

}

void loadAndPredict(){

    Dataset dataset("./../lightnet-ml/data/iris_flowers.csv", true);
    dataset.scale();

    Dataset testData = dataset.splitTestData(5);

    NNClassifier net = NNClassifier::loadModel("./../lightnet-ml/data/model.json");

    std::vector<NNClassifier::Prediction> predictions = net.predict(testData);

    for(NNClassifier::Prediction prediction : predictions){
        cout << "Predicted: " << prediction.predictedEncodedTarget << " Actual: " << prediction.actualEncodedTarget << " Conf: " << prediction.confidence << endl;
    }

}
