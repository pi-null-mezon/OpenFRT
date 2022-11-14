#include "illumestimator.h"

#include <opencv2/imgproc.hpp>

namespace cv { namespace ofrt {

IllumEstimator::IllumEstimator(const std::string &modelfilename) :
    FaceClassifier(cv::Size(120,120),45.0f,-0.1f)
{    
}

float nonuniformity(std::vector<float> values) // 1.0 - max nonuniformity, 0.0 - min nonuniformity
{
    const auto result = std::minmax_element(values.begin(),values.end());
    return (*result.second - *result.first) / 255.0f;
}

float mean(std::vector<float> values)
{
    if(values.size() > 0)
        return std::accumulate(values.begin(),values.end(),0.0f) / values.size();
    return 0.0f;
}

float estimateContrast(const cv::Mat &bgrmat) // 1.0 - max contrast, 0.0 - min contrast
{
    cv::Scalar means, stds;
    cv::Mat hlsmat;
    cv::cvtColor(bgrmat,hlsmat,cv::COLOR_BGR2HLS);
    cv::meanStdDev(hlsmat,means,stds);
    if(stds[2] > 0)
        return std::min(1.0, (3 * (stds[1] + stds[2]) / 2) / 255.0);
    return std::min(1.0, 2.3 * stds[1] / 255.0);
}

bool isGreyScale(const cv::Mat &mat, int stride)
{
    if(mat.channels() == 3) {
        const unsigned char *pix;
        unsigned char R, G, B;
        for(int y = 0; y < mat.rows; y = y + stride) {
            pix = mat.ptr(y);
            for(int x = 0; x < mat.cols; x = x + stride) {
                B = pix[3*x];
                G = pix[3*x+1];
                R = pix[3*x+2];
                if(std::max({std::abs((int)B - (int)G),
                              std::abs((int)B - (int)R),
                              std::abs((int)R - (int)G)}) > 2) {
                    return false;
                }
            }
        }
    }
    return true;
}

std::vector<cv::Point2f> transformCoordinates(const cv::Mat &rmatrix, const std::vector<cv::Point2f> &_landmarks)
{
    std::vector<cv::Point2f> pts(_landmarks.size(),cv::Point2f(0,0));
    for(size_t i=0; i < _landmarks.size(); ++i) {
        pts[i].x = _landmarks[i].x * rmatrix.at<double>(0,0) + _landmarks[i].y * rmatrix.at<double>(0,1) + rmatrix.at<double>(0,2);
        pts[i].y = _landmarks[i].x * rmatrix.at<double>(1,0) + _landmarks[i].y * rmatrix.at<double>(1,1) + rmatrix.at<double>(1,2);
    }
    return pts;
}

std::vector<std::vector<cv::Point>> facePartsPolygons(const std::vector<cv::Point2f> &_landmarks, bool include_whole_face)
{
    static const uint8_t _reye[] = {36,37,38,39,40,41};
    static const uint8_t _leye[] = {42,43,44,45,46,47};

    std::vector<cv::Point2f> points;
    if(_landmarks.size() == 68) {
        std::vector<cv::Point2f> rpts;
        rpts.reserve(74);

        cv::Point2f _rc(0,0), _lc(0,0);
        int _len = sizeof(_reye)/sizeof(_reye[0]);
        for(int i = 0; i < _len; ++i) {
            _rc += _landmarks[_reye[i]];
            _lc += _landmarks[_leye[i]];
        }
        _rc /= _len;
        _lc /= _len;
        cv::Point2f _cd = _lc - _rc;
        float _angle = 180.0f * static_cast<float>(std::atan(_cd.y/_cd.x) / CV_PI);
        cv::Point2f _cp = (_rc + _lc)/2.0f;
        cv::Mat _tm = cv::getRotationMatrix2D(_cp,_angle,1.0);
        for(size_t i=0; i < _landmarks.size(); ++i) {
            const cv::Point2f &_pt = _landmarks[i];
            rpts.push_back(cv::Point2f(_pt.x*_tm.at<double>(0,0) + _pt.y*_tm.at<double>(0,1) + _tm.at<double>(0,2),
                                       _pt.x*_tm.at<double>(1,0) + _pt.y*_tm.at<double>(1,1) + _tm.at<double>(1,2)));
        }
        // Additional points to cover forehead above eyes
        float _height = std::abs(0.4f*(rpts[0] - rpts[3] + rpts[16] - rpts[13]).y);
        rpts.push_back(cv::Point2f((rpts[0].x+rpts[17].x)/2.0f,(rpts[0].y+rpts[17].y)/2.0f - _height/2.0f));
        rpts.push_back(cv::Point2f(rpts[17].x,rpts[17].y - _height/1.25f));
        rpts.push_back(cv::Point2f(rpts[19].x,rpts[19].y - _height));
        rpts.push_back(cv::Point2f(rpts[24].x,rpts[24].y - _height));
        rpts.push_back(cv::Point2f(rpts[26].x,rpts[26].y - _height/1.25f));
        rpts.push_back(cv::Point2f((rpts[26].x+rpts[16].x)/2.0f,(rpts[26].y+rpts[16].y)/2.0f - _height/2.0f));

        // Unwrap
        points.reserve(74);
        _tm = cv::getRotationMatrix2D(_cp,-_angle,1.0);
        for(size_t i=0; i < rpts.size(); ++i) {
            const cv::Point2f &_pt = rpts[i];
            points.push_back(cv::Point2f(_pt.x*_tm.at<double>(0,0) + _pt.y*_tm.at<double>(0,1) + _tm.at<double>(0,2),
                                         _pt.x*_tm.at<double>(1,0) + _pt.y*_tm.at<double>(1,1) + _tm.at<double>(1,2)));
        }
    }

    std::vector<std::vector<cv::Point>> parts;
    if(points.size() == 74) {
        const float _yg = (points[8] - points[70]).y / 14.0f;
        const float _xg = (points[16] - points[0]).x / 18.f;

        /*const cv::Point2f ftc = (points[71]+points[70])/2.0f;
        std::vector<cv::Point> _rightforehead;
        for(size_t i = 22; i <= 25; ++i)
            _rightforehead.push_back(cv::Point2f(points[i].x,points[i].y - _yg/2.0f));
        _rightforehead.push_back(points[72]);
        _rightforehead.push_back(points[71]);
        _rightforehead.push_back(cv::Point2f(ftc.x + _xg/4.0f,ftc.y));
        parts.push_back(std::move(_rightforehead));*/

        std::vector<cv::Point> _rightcheek;
        for(size_t i = 15; i >= 12; --i)
            _rightcheek.push_back(cv::Point2f(points[i].x - _xg,points[i].y));
        _rightcheek.push_back(cv::Point2f(points[25].x,points[45].y + _yg));
        parts.push_back(std::move(_rightcheek));

        std::vector<cv::Point> _rightpart;
        _rightpart.push_back((cv::Point2f(points[12].x - _xg,points[12].y) + cv::Point2f(points[54].x,points[54].y - _yg)) / 2.0f);
        _rightpart.push_back(cv::Point2f(points[54].x,points[54].y - _yg));
        _rightpart.push_back(cv::Point2f(points[42].x,points[42].y + _yg));
        _rightpart.push_back(cv::Point2f(points[47].x,points[47].y + _yg));
        _rightpart.push_back(cv::Point2f(points[46].x - _xg,points[46].y + _yg));
        parts.push_back(std::move(_rightpart));

        std::vector<cv::Point> _nose;
        _nose.push_back(points[27]);
        const cv::Point2f shift = points[33] - points[30];
        for(size_t i = 0; i < 5; ++i)
            _nose.push_back(points[31 + i] - shift);
        parts.push_back(std::move(_nose));

        std::vector<cv::Point> _leftpart;
        _leftpart.push_back((cv::Point2f(points[4].x + _xg,points[4].y) + cv::Point2f(points[48].x,points[48].y - _yg)) / 2.0f);
        _leftpart.push_back(cv::Point2f(points[48].x,points[48].y - _yg));
        _leftpart.push_back(cv::Point2f(points[39].x,points[39].y + _yg));
        _leftpart.push_back(cv::Point2f(points[40].x,points[40].y + _yg));
        _leftpart.push_back(cv::Point2f(points[41].x + _xg,points[41].y + _yg));
        parts.push_back(std::move(_leftpart));

        std::vector<cv::Point> _leftcheek;
        for(size_t i = 1; i <= 4; ++i)
            _leftcheek.push_back(cv::Point2f(points[i].x + _xg,points[i].y));

        _leftcheek.push_back(cv::Point2f(points[18].x,points[36].y + _yg));
        parts.push_back(std::move(_leftcheek));

        /*std::vector<cv::Point> _leftrorehead;
        for(size_t i = 18; i <= 21; ++i)
            _leftrorehead.push_back(cv::Point2f(points[i].x,points[i].y - _yg/2.0f));
        _leftrorehead.push_back(cv::Point2f(ftc.x - _xg/4.0f,ftc.y));
        _leftrorehead.push_back(points[70]);
        _leftrorehead.push_back(points[69]);
        parts.push_back(std::move(_leftrorehead));*/

        if(include_whole_face) {
            std::vector<cv::Point> _wholeface;
            for(size_t i = 0; i < 17; ++i)
                _wholeface.push_back(points[i]);
            for(size_t i = 73; i > 68; --i)
                _wholeface.push_back(points[i]);
            parts.push_back(std::move(_wholeface));
        }
    }
    return parts;
}

std::vector<float> IllumEstimator::process(const cv::Mat &img, const std::vector<cv::Point2f> &landmarks, bool fast)
{  
    cv::Mat rm;
    const cv::Mat normalizedfacepatch = extractFacePatch(img,landmarks,iod(),size(),0,v2hshift(),true,cv::INTER_AREA,&rm);
    const std::vector<cv::Point2f> normalizedlandmarks = transformCoordinates(rm,landmarks);
    std::vector<std::vector<cv::Point>> polygons = facePartsPolygons(normalizedlandmarks,false);
    std::vector<float> vCH1(polygons.size(),0), vCH2(polygons.size(),0), vCH3(polygons.size(),0);
    for(size_t i = 0; i < polygons.size(); ++i) {
        cv::Mat mask = cv::Mat::zeros(normalizedfacepatch.rows,normalizedfacepatch.cols,CV_8UC1);
        cv::fillPoly(mask,polygons[i],cv::Scalar(255));
        const cv::Scalar meancolor = cv::mean(normalizedfacepatch,mask);
        vCH1[i] = meancolor[0];
        vCH2[i] = meancolor[1];
        vCH3[i] = meancolor[2];
    }
    std::vector<float> features(4,0);
    features[0] = isGreyScale(normalizedfacepatch,fast ? 4 : 2); // same colors in all channels
    features[1] = estimateContrast(normalizedfacepatch); // contrast
    features[2] = std::max({nonuniformity(vCH1),nonuniformity(vCH2),nonuniformity(vCH3)}); // non uniformity
    features[3] = ((mean(vCH1) + mean(vCH2) + mean(vCH3)) / 3.0f) / 255.0f; // exposure
    return features;
}

Ptr<FaceClassifier> IllumEstimator::createClassifier(const std::string &modelfilename)
{
    return makePtr<IllumEstimator>(modelfilename);
}

}}
