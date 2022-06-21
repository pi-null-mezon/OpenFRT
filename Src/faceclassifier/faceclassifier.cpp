#include "faceclassifier.h"

#include <opencv2/imgproc.hpp>

namespace cv { namespace ofrt {

FaceClassifier::FaceClassifier(const cv::Size &_size, float _iod, float _v2hshift):
    m_size(_size),
    m_iod(_iod),
    m_v2hshift(_v2hshift)
{
}

FaceClassifier::~FaceClassifier()
{

}

const cv::Size FaceClassifier::size() const
{
    return m_size;
}

float FaceClassifier::iod() const
{
    return m_iod;
}

float FaceClassifier::v2hshift() const
{
    return m_v2hshift;
}

dlib::matrix<dlib::rgb_pixel> FaceClassifier::cvmat2dlibmatrix(const cv::Mat &_cvmat)
{
    cv::Mat _mat = _cvmat;
    if(_cvmat.isContinuous() == false)
        _mat = _cvmat.clone();
    unsigned char *_p = _mat.ptr<unsigned char>(0);
    dlib::matrix<dlib::rgb_pixel> _img(_mat.rows,_mat.cols);
    for(long i = 0; i < static_cast<long>(_mat.total()); ++i)
        _img(i) = dlib::rgb_pixel(_p[3*i+2],_p[3*i+1],_p[3*i]); // BGR to RGB
    return _img;
}

cv::Mat FaceClassifier::extractFacePatch(const cv::Mat &_rgbmat, const std::vector<cv::Point2f> &_landmarks, float _targeteyesdistance, const cv::Size &_targetsize, float h2wshift, float v2hshift, bool rotate, int _interpolationtype)
{
    static uint8_t _reye[] = {36,37,38,39,40,41};
    static uint8_t _leye[] = {42,43,44,45,46,47};

    cv::Mat _patch;
    cv::Point2f _rc(0,0), _lc(0,0);
    if(_landmarks.size() == 68) {
        int _len = sizeof(_reye)/sizeof(_reye[0]);
        for(int i = 0; i < _len; ++i) {
            _rc += _landmarks[_reye[i]];
            _lc += _landmarks[_leye[i]];
        }
        _rc /= _len;
        _lc /= _len;
    } else if (_landmarks.size() == 5) {
        _rc = _landmarks[0];
        _lc = _landmarks[1];
    }
    cv::Point2f _cd = _lc - _rc;
    float _eyesdistance = std::sqrt((_cd.x)*(_cd.x) + (_cd.y)*(_cd.y));
    float _scale = _targeteyesdistance / _eyesdistance;
    float _angle = rotate ? 180.0f * static_cast<float>(std::atan(_cd.y/_cd.x) / CV_PI) : 0;
    cv::Point2f _cp = (_rc + _lc)/2.0f;
    cv::Mat _tm = cv::getRotationMatrix2D(_cp,_angle,_scale);
    _tm.at<double>(0,2) += _targetsize.width/2.0 - _cp.x + h2wshift * _targetsize.width;
    _tm.at<double>(1,2) += _targetsize.height/2.0 - _cp.y + v2hshift * _targetsize.height;

    cv::warpAffine(_rgbmat,_patch,_tm,_targetsize,_interpolationtype);

    return _patch;
}

}}
