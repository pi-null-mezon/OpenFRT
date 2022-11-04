#ifndef FACEMARK_H
#define FACEMARK_H

#include <opencv2/core.hpp>

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
};

}}

#endif // FACEMARK_H
