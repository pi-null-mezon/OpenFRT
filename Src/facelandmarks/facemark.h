#ifndef FACEMARK_H
#define FACEMARK_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace cv { namespace ofrt {

/**
 * @brief The Facemark is a base abstract class for all face landmarking algorithms
 */
class Facemark
{
public:
    Facemark();

    virtual ~Facemark();

    virtual bool fit(const cv::Mat &image,
                     const std::vector<Rect> &faces,
                     std::vector<std::vector<Point2f>> &landmarks) const = 0;

    static cv::Rect prepareRect(const cv::Rect &source, const cv::Rect &frame, float upscale);

    static cv::Mat extractFace(const cv::Mat &_rgbmat, const std::vector<cv::Point2f> &_landmarks, float _targeteyesdistance, const cv::Size &_targetsize, float h2wshift, float v2hshift, bool rotate, int interpolationtype=cv::INTER_LINEAR);
};

}}

#endif // FACEMARK_H
