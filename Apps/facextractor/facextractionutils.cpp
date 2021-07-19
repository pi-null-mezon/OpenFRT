#include "facextractionutils.h"

#include <opencv2/imgproc.hpp>

cv::Rect squareRectFromCenter(const cv::Rect &_source)
{
    if(_source.width == _source.height)
        return _source;
    else if(_source.width > _source.height)
        return cv::Rect(_source.x + (_source.width - _source.height) / 2, _source.y,_source.height, _source.height);
    return cv::Rect(_source.x, _source.y + (_source.height - _source.width) / 2, _source.width, _source.width);
}

cv::Mat extractFacePatch(const cv::Mat &_rgbmat, const std::vector<cv::Point2f> &_landmarks, float _targeteyesdistance, const cv::Size &_targetsize, float h2wshift, float v2hshift)
{
    static uint8_t _reye[] = {36,37,38,39,40,41};
    static uint8_t _leye[] = {42,43,44,45,46,47};

    cv::Mat _patch;
    if(_landmarks.size() == 68) {
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
        float _angle = 180.0f * static_cast<float>(std::atan(_cd.y/_cd.x) / CV_PI);
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

std::vector<std::vector<cv::Point2f>> detectFacesLandmarks(const cv::Mat &_rgbmat, cv::Ptr<cv::ofrt::FaceDetector> &facedetector, cv::Ptr<cv::face::Facemark> &facelandmarker)
{
    std::vector<std::vector<cv::Point2f>> _vlandmarks;
    const std::vector<cv::Rect> _facesboxes = facedetector->detectFaces(_rgbmat);
    const float _upscalefactor = 1.4f;
    const cv::Rect framerect(0,0,_rgbmat.cols,_rgbmat.rows);
    std::vector<cv::Rect> _vfacesrects;
    _vfacesrects.reserve(_facesboxes.size());
    for(const cv::Rect &_facebox: _facesboxes) {
        const cv::Rect _facerect = squareRectFromCenter(_facebox);
        cv::Rect _facerectforlandmarks = cv::Rect(_facerect.x - _facerect.width * (_upscalefactor - 1.0f) / 2.0f,
                                                  _facerect.y - _facerect.height * (_upscalefactor - 1.0f) / 2.0f,
                                                  _facerect.width*_upscalefactor,
                                                  _facerect.height*_upscalefactor) & framerect;
        _vfacesrects.push_back(std::move(_facerectforlandmarks));
    }
    facelandmarker->fit(_rgbmat,_vfacesrects,_vlandmarks);
    return _vlandmarks;
}
