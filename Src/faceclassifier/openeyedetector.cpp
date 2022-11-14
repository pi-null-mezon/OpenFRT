#include "openeyedetector.h"

#include <opencv2/imgproc.hpp>

namespace cv { namespace ofrt {

OpenEyeDetector::OpenEyeDetector(const std::string &modelfilename) :
    FaceClassifier(cv::Size(240,320),80.0f,0)
{
    try {
        dlib::blink_net_type net;
        dlib::deserialize(modelfilename) >> net;
        snet.subnet() = net.subnet();
    } catch(const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
}

std::vector<float> OpenEyeDetector::process(const Mat &img, const std::vector<Point2f> &landmarks, bool fast)
{
    const std::vector<cv::Mat> patches = extractEyesPatches(img,landmarks,iod(),size());
    std::vector<float> openeyes(patches.size(),0);
    for(size_t i = 0 ; i < patches.size(); ++i) {
        cv::Mat normalizedpatch;
        cv::resize(patches[i],normalizedpatch,cv::Size(48,48));
        std::vector<dlib::matrix<dlib::rgb_pixel>> crops(1,cvmat2dlibmatrix(normalizedpatch));
        if(!fast)
            crops.emplace_back(dlib::fliplr(crops[0]));
        dlib::matrix<float,1,2> p = dlib::sum_rows(dlib::mat(snet(crops.begin(),crops.end())))/crops.size();
        openeyes[i] = p(1);
    }
    return openeyes; // 0 - right, 1 - left
}

cv::Ptr<FaceClassifier> OpenEyeDetector::createClassifier(const std::string &modelfilename)
{
    return makePtr<OpenEyeDetector>(modelfilename);
}

std::vector<cv::Mat> OpenEyeDetector::extractEyesPatches(const cv::Mat &_rgbmat, const std::vector<cv::Point2f> &_landmarks, float _targeteyesdistance, const cv::Size &_targetsize)
{
    std::vector<cv::Mat> patches;
    patches.reserve(2);
    if(_landmarks.size() == 68) {
        static const uint8_t reye[] = {36,37,38,39,40,41};
        static const uint8_t leye[] = {42,43,44,45,46,47};
        cv::Point2f rc(0,0), lc(0,0);
        int len = sizeof(reye)/sizeof(reye[0]);
        for(int i = 0; i < len; ++i) {
            rc += _landmarks[reye[i]];
            lc += _landmarks[leye[i]];
        }
        rc /= len;
        lc /= len;

        cv::Point2f cd = lc - rc;
        float _eyesdistance = std::sqrt((cd.x)*(cd.x) + (cd.y)*(cd.y));
        float _scale = _targeteyesdistance / _eyesdistance;
        float _angle = 180.0f * static_cast<float>(std::atan(cd.y/cd.x) / CV_PI);
        cv::Point2f _cp = (rc + lc)/2.0f;
        cv::Mat _tm = cv::getRotationMatrix2D(_cp,_angle,_scale);
        _tm.at<double>(0,2) += _targetsize.width/2.0 - _cp.x;
        _tm.at<double>(1,2) += _targetsize.height/2.0 - _cp.y;

        int _interpolationtype = cv::INTER_LINEAR;
        if(_scale < 1.0f)
            _interpolationtype = cv::INTER_AREA;
        cv::Mat _normalizedfaceimg;
        cv::warpAffine(_rgbmat,_normalizedfaceimg,_tm,_targetsize,_interpolationtype);
        const cv::Rect imgrect(0,0,_normalizedfaceimg.cols,_normalizedfaceimg.rows);
        // We should calculate center points in the new CS
        cv::Point2d nrc(rc.x*_tm.at<double>(0,0) + rc.y*_tm.at<double>(0,1) + _tm.at<double>(0,2),
                         rc.x*_tm.at<double>(1,0) + rc.y*_tm.at<double>(1,1) + _tm.at<double>(1,2));
        cv::Point2d nlc(lc.x*_tm.at<double>(0,0) + lc.y*_tm.at<double>(0,1) + _tm.at<double>(0,2),
                         lc.x*_tm.at<double>(1,0) + lc.y*_tm.at<double>(1,1) + _tm.at<double>(1,2));

        cv::RotatedRect rotatedrect(nrc,cv::Size(_targetsize.width/5,_targetsize.width/5),0);
        cv::Rect roirect = rotatedrect.boundingRect() & imgrect;
        patches.push_back(_normalizedfaceimg(roirect));

        rotatedrect.center = nlc;
        roirect = rotatedrect.boundingRect() & imgrect;
        patches.push_back(_normalizedfaceimg(roirect));
    }
    return patches;
}

}}

