#include "headposepredictor.h"

#include <opencv2/imgproc.hpp>

namespace cv { namespace ofrt {

HeadPosePredictor::HeadPosePredictor(const std::string &modelfilename) :
    FaceClassifier(cv::Size(200,200),45.0f,0)
{
    try {
        dlib::deserialize(modelfilename) >> net;
    } catch(const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
}

std::vector<float> HeadPosePredictor::classify(const Mat &img, const std::vector<Point2f> &landmarks)
{
    cv::Mat cv_facepatch = extractFacePatch(img,landmarks,iod(),size(),0,v2hshift(),false,cv::INTER_LINEAR);
    cv::resize(cv_facepatch,cv_facepatch,cv::Size(80,80),0,0,cv::INTER_AREA);
    std::vector<dlib::matrix<dlib::rgb_pixel>> crops(2,dlib::matrix<dlib::rgb_pixel>());
    crops[0] = cvmat2dlibmatrix(cv_facepatch);
    crops[1] = dlib::fliplr(crops[0]);
    std::vector<dlib::matrix<float>> p = net(crops);
    std::vector<float> angles(3,0);
    angles[0] = -90.0f * (p[0](0) - p[1](0)) / 2.0f; // yaw
    angles[1] = -90.0f * (p[0](1) + p[1](1)) / 2.0f; // pitch
    angles[2] = -90.0f * (p[0](2) - p[1](2)) / 2.0f; // roll
    return angles;
}

float HeadPosePredictor::confidence(const Mat &img, const std::vector<Point2f> &landmarks)
{
    return classify(img,landmarks)[0];
}

Ptr<FaceClassifier> HeadPosePredictor::createClassifier(const std::string &modelfilename)
{
    return makePtr<HeadPosePredictor>(modelfilename);
}

}}
