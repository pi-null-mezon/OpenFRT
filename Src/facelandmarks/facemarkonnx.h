#ifndef FACEMARKONNX_H
#define FACEMARKONNX_H

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include "facemark.h"

namespace cv { namespace ofrt {

/**
 * @brief The FacemarkCNN class is a custom 68 facial points detector based on CNN
 */
class FacemarkONNX : public Facemark {

public:
    FacemarkONNX(const String &modelfilename);

    bool fit(const cv::Mat &image,
             const std::vector<Rect> &faces,
             std::vector<std::vector<Point2f>> &landmarks) const override;

    static Ptr<Facemark> create(const String &modelfilename);

private:
    mutable cv::dnn::Net net;
    cv::Size isize;
};

}}

#endif // FACEMARKCNN_H
