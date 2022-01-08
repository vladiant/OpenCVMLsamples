// https://titanwolf.org/Network/Articles/Article?AID=5d4c85d5-8813-4d7c-b4f9-cf0f4a829d74

#include <iostream>
#include <opencv2/ml/ml.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
  float trainingData[42][2] = {
      {40, 55},  {35, 35},  {55, 15},  {45, 25},  {10, 10}, {15, 15}, {40, 10},
      {30, 15},  {30, 50},  {100, 20}, {45, 65},  {20, 35}, {80, 20}, {90, 5},
      {95, 35},  {80, 65},  {15, 55},  {25, 65},  {85, 35}, {85, 55}, {95, 70},
      {105, 50}, {115, 65}, {110, 25}, {120, 45}, {15, 45}, {55, 30}, {60, 65},
      {95, 60},  {25, 40},  {75, 45},  {105, 35}, {65, 10}, {50, 50}, {40, 35},
      {70, 55},  {80, 30},  {95, 45},  {60, 20},  {70, 30}, {65, 45}, {85, 40}};
  Mat trainingDataMat(42, 2, CV_32FC1, trainingData);

  int32_t responses[42] = {
      'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
      'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'B', 'B',
      'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'};
  Mat responsesMat(42, 1, CV_32SC1, responses);

  float priors[2] = {1, 1};

  auto boost = cv::ml::Boost::create();
  boost->setBoostType(cv::ml::Boost::REAL);
  boost->setWeakCount(10);
  boost->setWeightTrimRate(0.95);
  boost->setUseSurrogates(false);
  boost->setPriors(Mat(2, 1, CV_32FC1, priors));

  boost->train(trainingDataMat, cv::ml::ROW_SAMPLE, responsesMat);
  float myData[2] = {55, 25};
  Mat myDataMat(2, 1, CV_32FC1, myData);
  double r = boost->predict(myDataMat);

  cout << "result:  " << (char)r << endl;

  return 0;
}