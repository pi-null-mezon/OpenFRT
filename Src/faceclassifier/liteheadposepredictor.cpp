#include "liteheadposepredictor.h"

#include <opencv2/imgproc.hpp>

namespace cv { namespace ofrt {

LiteHeadPosePredictor::LiteHeadPosePredictor(const std::string &modelfilename) :
    FaceClassifier(cv::Size(80,80),18.0f,0)
{
    try {
        dlib::deserialize(modelfilename) >> net;
    } catch(const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
}

/*cv::Mat LiteHeadPosePredictor::crop(const Mat &img, const std::vector<Point2f> &landmarks, const cv::Size targetsize)
{
    assert(landmarks.size() == 68);
    const cv::Rect rect = scale_rect(cv::boundingRect(landmarks),1.8f);
    const cv::Point2f &_cp = landmarks[28];
    const float scale = (float)targetsize.width / rect.width;
    cv::Mat _tm = cv::getRotationMatrix2D(_cp,0,scale);
    _tm.at<double>(0,2) += targetsize.width/2.0 - _cp.x;
    _tm.at<double>(1,2) += targetsize.height/2.0 - _cp.y;
    cv::Mat normalizedmat;
    cv::warpAffine(img,normalizedmat,_tm,targetsize,cv::INTER_LINEAR);
    return normalizedmat;
}*/

std::vector<float> LiteHeadPosePredictor::process(const Mat &img, const std::vector<Point2f> &landmarks, bool fast)
{
    cv::Mat cv_facepatch = extractFacePatch(img,landmarks,iod(),size(),0,v2hshift(),false,cv::INTER_LINEAR);
    std::vector<dlib::matrix<dlib::rgb_pixel>> crops(1,cvmat2dlibmatrix(cv_facepatch));
    if(!fast)
        crops.emplace_back(dlib::fliplr(crops[0]));
    std::vector<dlib::matrix<float>> p = net(crops);
    std::vector<float> angles(3,0);
    if(crops.size() == 2) {
        angles[0] = -90.0f * (p[0](0) - p[1](0)) / 2.0f; // yaw
        angles[1] = -90.0f * (p[0](1) + p[1](1)) / 2.0f; // pitch
        angles[2] = -90.0f * (p[0](2) - p[1](2)) / 2.0f; // roll
    } else {
        angles[0] = -90.0f * p[0](0); // yaw
        angles[1] = -90.0f * p[0](1); // pitch
        angles[2] = -90.0f * p[0](2); // roll
    }
    return angles;
}

Ptr<FaceClassifier> LiteHeadPosePredictor::createClassifier(const std::string &modelfilename)
{
    return makePtr<LiteHeadPosePredictor>(modelfilename);
}

}}
