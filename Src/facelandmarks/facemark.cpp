#include "facemark.h"

#include <opencv2/imgproc.hpp>

namespace cv { namespace ofrt {

Facemark::Facemark()
{
}

Facemark::~Facemark()
{
}

Rect Facemark::prepareRect(const cv::Rect &source, const cv::Rect &frame, float upscale)
{
    cv::Rect rect;
    if(source.width == source.height)
        rect = source;
    else if(source.width > source.height)
        rect = cv::Rect(source.x + (source.width - source.height) / 2, source.y, source.height, source.height);
    else
        rect = cv::Rect(source.x, source.y + (source.height - source.width) / 2, source.width, source.width);
    return (cv::Rect(rect.x - rect.width * (upscale - 1.0f) / 2.0f,
                     rect.y - rect.height * (upscale - 1.0f) / 2.0f,
                     rect.width*upscale,rect.height*upscale) & frame);
}

Mat Facemark::extractFace(const Mat &_rgbmat, const std::vector<Point2f> &_landmarks, float _targeteyesdistance, const Size &_targetsize, float h2wshift, float v2hshift, bool rotate)
{
    cv::Mat _patch;
    if(_landmarks.size() == 68) {
        static uint8_t _reye[] = {36,37,38,39,40,41};
        static uint8_t _leye[] = {42,43,44,45,46,47};
        cv::Point2f _rc(0,0), _lc(0,0);
        int _len = sizeof(_reye)/sizeof(_reye[0]);
        for(int i = 0; i < _len; ++i) {
            _rc += _landmarks[_reye[i]];
            _lc += _landmarks[_leye[i]];
        }
        _rc /= _len;
        _lc /= _len;
        cv::Point2f _cd = _lc - _rc;
        float _eyesdistance = std::sqrt((_cd.x)*(_cd.x) + (_cd.y)*(_cd.y));
        float _scale = _targeteyesdistance / _eyesdistance;
        float _angle = rotate ? 180.0f * static_cast<float>(std::atan(_cd.y/_cd.x) / CV_PI) : 0;
        cv::Point2f _cp = (_rc + _lc)/2.0f;
        cv::Mat _tm = cv::getRotationMatrix2D(_cp,_angle,_scale);
        _tm.at<double>(0,2) += _targetsize.width/2.0 - _cp.x + h2wshift * _targetsize.width;
        _tm.at<double>(1,2) += _targetsize.height/2.0 - _cp.y + v2hshift * _targetsize.height;

        int _interpolationtype = cv::INTER_LINEAR;
        if(_scale < 1)
            _interpolationtype = cv::INTER_AREA;
        cv::warpAffine(_rgbmat,_patch,_tm,_targetsize,_interpolationtype);
    }
    return _patch;
}

}}
