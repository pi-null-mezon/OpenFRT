#ifndef ROTATECLASSIFIER_H
#define ROTATECLASSIFIER_H

#include <opencv2/dnn.hpp>

#include "faceclassifier.h"

namespace cv { namespace ofrt {

/**
 * @brief The FacemarkCNN class is a custom 68 facial points detector based on CNN
 */
class RotateClassifier : public FaceClassifier {

public:
    RotateClassifier(const String &modelfilename);

    std::vector<float> process(const cv::Mat &img, const std::vector<cv::Point2f> &landmarks, bool fast) override;

    std::vector<float> process(const cv::Mat &img, const cv::Rect facerect, bool fast);

    static Ptr<FaceClassifier> createClassifier(const String &modelfilename="./rotate_net.onnx");

private:
    mutable cv::dnn::Net net;
    std::vector<String> output_names;
};

}}

#endif // ROTATECLASSIFIER_H
