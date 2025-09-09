#ifndef FACELANDMARKEROV2_H
#define FACELANDMARKEROV2_H

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include "facemark.h"

namespace cv { namespace ofrt {

class FaceLandmarkerOV2 : public Facemark {

public:
    FaceLandmarkerOV2(const String &modelfilename);

    bool fit(const cv::Mat &image,
             const std::vector<Rect> &faces,
             std::vector<std::vector<Point2f>> &landmarks) const override;

    std::vector<std::vector<float>> getConfidences() const;

    static Ptr<Facemark> create(const String &modelfilename="./onet_ea_0004_13.onnx");

private:
    mutable cv::dnn::Net net;
    cv::Size isize;
    std::vector<String> output_names;
    mutable std::vector<std::vector<float>> confidences;
};

}}

#endif // FACELANDMARKEROV2_H
