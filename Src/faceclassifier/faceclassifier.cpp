#include "faceclassifier.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

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

/*dlib::matrix<dlib::rgb_pixel> FaceClassifier::cvmat2dlibmatrix(const cv::Mat &_cvmat)
{
    cv::Mat _mat = _cvmat;
    if(_cvmat.isContinuous() == false)
        _mat = _cvmat.clone();
    unsigned char *_p = _mat.ptr<unsigned char>(0);
    dlib::matrix<dlib::rgb_pixel> _img(_mat.rows,_mat.cols);
    for(long i = 0; i < static_cast<long>(_mat.total()); ++i)
        _img(i) = dlib::rgb_pixel(_p[3*i+2],_p[3*i+1],_p[3*i]); // BGR to RGB
    return _img;
}*/

cv::Mat FaceClassifier::extractFacePatch(const cv::Mat &_rgbmat, const std::vector<cv::Point2f> &_landmarks, float _targeteyesdistance, const cv::Size &_targetsize, float h2wshift, float v2hshift, bool rotate, int _interpolationtype, cv::Mat *rmatrix)
{ 
    static uint8_t _reye[] = {36,37,38,39,40,41};
    static uint8_t _leye[] = {42,43,44,45,46,47};
    cv::Mat _patch;
    cv::Point2f _rc(0,0), _lc(0,0), nose(0,0);
    if(_landmarks.size() == 68) {
        int _len = sizeof(_reye)/sizeof(_reye[0]);
        for(int i = 0; i < _len; ++i) {
            _rc += _landmarks[_reye[i]];
            _lc += _landmarks[_leye[i]];
        }
        _rc /= _len;
        _lc /= _len;
        nose = _landmarks[27] - _landmarks[33];
    } else if (_landmarks.size() == 5) {
        _rc = _landmarks[0];
        _lc = _landmarks[1];
        nose = (_rc + _lc) / 2 - _landmarks[2];
    }
    cv::Point2f _cd = _lc - _rc;
    float _eyesdistance = std::sqrt((_cd.x)*(_cd.x) + (_cd.y)*(_cd.y));
    float _noselength = std::sqrt((nose.x)*(nose.x) + (nose.y)*(nose.y)) + 1E-6;
    float _scale = _targeteyesdistance / _eyesdistance;
    float _angle = rotate ? 180.0f * static_cast<float>(std::atan(_cd.y/_cd.x) / CV_PI) : 0;
    if(_eyesdistance / _noselength  < 1.0f) { // very high yaw
        _scale = _targeteyesdistance / (1.4 * _noselength);
        _angle = 0;
    }
    cv::Point2f _cp = (_rc + _lc) / 2.0f;
    cv::Mat _tm = cv::getRotationMatrix2D(_cp,_angle,_scale);
    _tm.at<double>(0,2) += _targetsize.width / 2.0 - _cp.x + h2wshift * _targetsize.width;
    _tm.at<double>(1,2) += _targetsize.height / 2.0 - _cp.y + v2hshift * _targetsize.height;
    if(rmatrix != nullptr)
        *rmatrix = _tm;
    cv::warpAffine(_rgbmat,_patch,_tm,_targetsize,_interpolationtype);
    return _patch;
}

Mat FaceClassifier::extractFacePatch(const Mat &_rgbmat, const std::vector<Point2f> &_landmarks, const Size &_targetsize, int _interpolationtype, Mat *rmatrix)
{
    std::vector<cv::Point2f> from;
    if(_landmarks.size() == 68) {
        from = std::vector<cv::Point2f>(5, cv::Point2f(0,0));
        static uint8_t _reye[] = {36,37,38,39,40,41};
        static uint8_t _leye[] = {42,43,44,45,46,47};
        int _len = sizeof(_reye)/sizeof(_reye[0]);
        for(int i = 0; i < _len; ++i) {
            from[0] += _landmarks[_reye[i]];
            from[1] += _landmarks[_leye[i]];
        }
        from[0] /= _len;
        from[1] /= _len;
        from[2] = _landmarks[33];
        from[3] = _landmarks[60];
        from[4] = _landmarks[64];
    } else if (_landmarks.size() == 5) {
        from[0] = _landmarks[0];
    }
    std::vector<cv::Point2f> to = {cv::Point2f(30.2946, 51.6963),
                                   cv::Point2f(65.5318, 51.5014),
                                   cv::Point2f(48.0252, 71.7366),
                                   cv::Point2f(33.5493, 92.3655),
                                   cv::Point2f(62.7299, 92.2041)};
    float scale_x = static_cast<float>(_targetsize.width) / 96.0;
    float scale_y = static_cast<float>(_targetsize.height) / 112.0;
    for(size_t i = 0; i < to.size(); ++i)
        to[i] = cv::Point2f(to[i].x*scale_x, to[i].y*scale_y);

    cv::Mat tm = cv::estimateAffinePartial2D(from, to, cv::noArray(), cv::LMEDS);
    if(rmatrix != nullptr)
        *rmatrix = tm;

    cv::Mat patch;
    cv::warpAffine(_rgbmat,patch,tm,_targetsize,_interpolationtype);
    return patch;
}


Mat FaceClassifier::extractFacePatch(const Mat &_rgbmat, const Rect facerect, const Size &_targetsize, float scale, int _interpolationtype)
{
    cv::Mat facepatch = _rgbmat(scale_rect(facerect,scale) & cv::Rect(0,0,_rgbmat.cols,_rgbmat.rows));
    cv::Rect2f roiRect = cv::Rect2f(0,0,0,0);
    if(static_cast<float>(facepatch.cols)/facepatch.rows > static_cast<float>(_targetsize.width)/_targetsize.height) {
        roiRect.height = static_cast<float>(facepatch.rows);
        roiRect.width = facepatch.rows * static_cast<float>(_targetsize.width)/_targetsize.height;
        roiRect.x = (facepatch.cols - roiRect.width)/2.0f;
    } else {
        roiRect.width = static_cast<float>(facepatch.cols);
        roiRect.height = facepatch.cols * static_cast<float>(_targetsize.height)/_targetsize.width;
        roiRect.y = (facepatch.rows - roiRect.height)/2.0f;
    }
    roiRect &= cv::Rect2f(0.0f, 0.0f, static_cast<float>(facepatch.cols), static_cast<float>(facepatch.rows));
    cv::Mat output;
    if(roiRect.area() > 0)  {
        cv::Mat croppedImg(facepatch, roiRect);
        cv::resize(croppedImg, output, _targetsize, 0, 0, _interpolationtype);
    }
    return output;
}

cv::Rect FaceClassifier::scale_rect(const cv::Rect &rect, float scale) {
    return cv::Rect(rect.x - rect.width * (scale - 1.0f) / 2.0f,
                    rect.y - rect.height * (scale - 1.0f) / 2.0f,
                    rect.width * scale,
                    rect.height * scale);
}

std::vector<float> FaceClassifier::softmax(const std::vector<float> &logits)
{
    return softmax(logits.data(), logits.size());
}

std::vector<float> FaceClassifier::softmax(const float *logits, unsigned long size)
{
    float exp_sum = 0.0f;
    std::vector<float> exs(size,0.0f);
    for(size_t i = 0; i < size; ++i) {
        exs[i] = std::exp(logits[i]);
        exp_sum += exs[i];
    }
    std::vector<float> probs(size,0.0f);
    for(size_t i = 0; i < size; ++i)
        probs[i] = exs[i] / exp_sum;
    return probs;
}

}}
