#ifndef CRFIQAESTIMATOR_H
#define CRFIQAESTIMATOR_H

#include <opencv2/dnn.hpp>

#include "faceclassifier.h"

namespace cv { namespace ofrt {

class CRFIQAEstimator : public FaceClassifier {

public:
    CRFIQAEstimator(const String &modelfilename);

    // CRFIQA score of the face
    std::vector<float> process(const cv::Mat &img, const std::vector<cv::Point2f> &landmarks, bool fast) override;

    static Ptr<FaceClassifier> createClassifier(const String &modelfilename="./crfiqa_estimator.onnx");

private:
    mutable cv::dnn::Net net;
    std::vector<String> output_names;
};

}}

#endif // CRFIQAESTIMATOR_H
