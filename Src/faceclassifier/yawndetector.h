#ifndef YAWNDETECTOR_H
#define YAWNDETECTOR_H

#include <opencv2/dnn.hpp>

#include "faceclassifier.h"

namespace cv { namespace ofrt {

class YawnDetector : public FaceClassifier {

public:
    YawnDetector(const String &modelfilename);

    // prob that the face is under yawning phase
    std::vector<float> process(const cv::Mat &img, const std::vector<cv::Point2f> &landmarks, bool fast) override;

    static Ptr<FaceClassifier> createClassifier(const String &modelfilename="./yawn_net.onnx");

private:
    mutable cv::dnn::Net net;
    std::vector<String> output_names;
};

}}

#endif // YAWNDETECTOR_H
