#ifndef OPENEYEDETECTOR_H
#define OPENEYEDETECTOR_H

#include <opencv2/dnn.hpp>

#include "faceclassifier.h"

namespace cv { namespace ofrt {

class OpenEyeDetector : public FaceClassifier
{
public:
    OpenEyeDetector(const std::string &modelfilename);

    std::vector<float> process(const cv::Mat &img, const std::vector<cv::Point2f> &landmarks, bool fast) override;

    static cv::Ptr<FaceClassifier> createClassifier(const std::string &modelfilename="./blink_net.onnx");

    static std::vector<cv::Mat> extractEyesPatches(const cv::Mat &_rgbmat, const std::vector<cv::Point2f> &_landmarks, float _targeteyesdistance, const cv::Size &_targetsize);

private:   
    mutable cv::dnn::Net net;
    std::vector<String> output_names;
};

}}

#endif // GLASSESDETECTOR_H
