// https://stackoverflow.com/questions/37500713/opencv-image-recognition-setting-up-ann-mlp

#include <iomanip>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace ml;
using namespace std;

void print(Mat& mat, int prec) {
  for (int i = 0; i < mat.size().height; i++) {
    cout << "[";
    for (int j = 0; j < mat.size().width; j++) {
      cout << fixed << setw(2) << setprecision(prec) << mat.at<float>(i, j);
      if (j != mat.size().width - 1)
        cout << ", ";
      else
        cout << "]" << endl;
    }
  }
}

int main() {
  const int hiddenLayerSize = 4;
  float inputTrainingDataArray[4][2] = {
      {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
  Mat inputTrainingData = Mat(4, 2, CV_32F, inputTrainingDataArray);

  float outputTrainingDataArray[4][1] = {{0.0}, {1.0}, {1.0}, {0.0}};
  Mat outputTrainingData = Mat(4, 1, CV_32F, outputTrainingDataArray);

  Ptr<ANN_MLP> mlp = ANN_MLP::create();

  Mat layersSize = Mat(3, 1, CV_16U);
  layersSize.row(0) = Scalar(inputTrainingData.cols);
  layersSize.row(1) = Scalar(hiddenLayerSize);
  layersSize.row(2) = Scalar(outputTrainingData.cols);
  mlp->setLayerSizes(layersSize);

  mlp->setActivationFunction(ANN_MLP::ActivationFunctions::SIGMOID_SYM);

  TermCriteria termCrit =
      TermCriteria(TermCriteria::Type::COUNT + TermCriteria::Type::EPS,
                   100000000, 0.000000000000000001);
  mlp->setTermCriteria(termCrit);

  mlp->setTrainMethod(ANN_MLP::TrainingMethods::BACKPROP);

  Ptr<TrainData> trainingData = TrainData::create(
      inputTrainingData, SampleTypes::ROW_SAMPLE, outputTrainingData);

  mlp->train(trainingData
             /*, ANN_MLP::TrainFlags::UPDATE_WEIGHTS
             + ANN_MLP::TrainFlags::NO_INPUT_SCALE
             + ANN_MLP::TrainFlags::NO_OUTPUT_SCALE*/
  );

  for (int i = 0; i < inputTrainingData.rows; i++) {
    Mat sample =
        Mat(1, inputTrainingData.cols, CV_32F, inputTrainingDataArray[i]);
    Mat result;
    mlp->predict(sample, result);
    cout << sample << " -> ";  // << result << endl;
    print(result, 0);
    cout << endl;
  }

  return 0;
}
