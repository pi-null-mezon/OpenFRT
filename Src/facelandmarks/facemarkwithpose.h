#ifndef FACEMARKWITHPOSE_H
#define FACEMARKWITHPOSE_H

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include "facemark.h"

namespace cv { namespace ofrt {

/**
 * @brief The FacemarkCNN class is a custom 68 facial points detector based on CNN
 */
class FacemarkWithPose : public Facemark {

public:
    FacemarkWithPose(const String &modelfilename);

    bool fit(const cv::Mat &image,
             const std::vector<Rect> &faces,
             std::vector<std::vector<Point2f>> &landmarks) const override;

    std::vector<std::vector<float>> last_pose() const;

    bool fit(const cv::Mat &image,
             const std::vector<Rect> &faces,
             std::vector<std::vector<Point2f>> &landmarks, std::vector<std::vector<float>> &angles) const;

    static Ptr<Facemark> create(const String &modelfilename);

private:
    mutable cv::dnn::Net net;
    cv::Size isize;
    std::vector<String> output_names;
    mutable std::vector<std::vector<float>> poses;
};

}}

#endif // FACEMARKWITHPOSE_H
