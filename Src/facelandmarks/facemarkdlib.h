#ifndef FACEMARKDLIB_H
#define FACEMARKDLIB_H

#include <opencv2/core.hpp>

#include "facemark.h"

#include <dlib/image_processing/shape_predictor.h>

namespace cv { namespace ofrt {

/**
 * @brief The FacemarkDlib class is a wrap for the dlib's 68 (and 5 also) facial points detector
 */
class FacemarkDlib : public Facemark {

public:
    FacemarkDlib(const String &model);

    bool fit(const cv::Mat &image,
             const std::vector<Rect> &faces,
             std::vector<std::vector<Point2f>> &landmarks) const override;

    static Ptr<Facemark> create(const String &model);

private:
    mutable dlib::shape_predictor shapepredictor;
};



}}




#endif // FACEMARKDLIB_H
