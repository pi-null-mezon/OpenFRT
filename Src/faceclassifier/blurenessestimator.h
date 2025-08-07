#ifndef BLURENESSESTIMATOR_H
#define BLURENESSESTIMATOR_H

#include <opencv2/dnn.hpp>

#include "faceclassifier.h"

namespace cv { namespace ofrt {

class BlurenessEstimator : public FaceClassifier {

public:
    BlurenessEstimator(const String &modelfilename);

    std::vector<float> process(const cv::Mat &img, const std::vector<cv::Point2f> &landmarks, bool fast) override;

    static Ptr<FaceClassifier> createClassifier(const String &modelfilename="./macroblocks_estimator.onnx");

private:
    mutable cv::dnn::Net net;
    std::vector<String> output_names;
};

}}

#endif // CRFIQAESTIMATOR_H
