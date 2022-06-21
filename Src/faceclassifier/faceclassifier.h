#ifndef FACECLASSISIER_H
#define FACECLASSISIER_H

#include <opencv2/core.hpp>
#include <dlib/dnn.h>

namespace cv { namespace ofrt {

// Base abstract class for all classifiers that work with aligned faces
class FaceClassifier
{
public:
    FaceClassifier(const cv::Size &_size, float _iod, float _v2hshift);

    virtual ~FaceClassifier();

    dlib::matrix<dlib::rgb_pixel> cvmat2dlibmatrix(const cv::Mat &_cvmat);

    cv::Mat extractFacePatch(const cv::Mat &_rgbmat,
                             const std::vector<cv::Point2f> &_landmarks,
                             float _targeteyesdistance,
                             const cv::Size &_targetsize,
                             float h2wshift,
                             float v2hshift,
                             bool rotate,
                             int _interpolationtype);

    virtual std::vector<float> classify(const cv::Mat &img, const std::vector<cv::Point2f> &landmarks) = 0;

    virtual float confidence(const cv::Mat &img, const std::vector<cv::Point2f> &landmarks) = 0;

    const cv::Size size() const;

    float iod() const;

    float v2hshift() const;

private:
    cv::Size m_size;
    float m_iod;
    float m_v2hshift;
};

}}
#endif // FACECLASSISIER_H
