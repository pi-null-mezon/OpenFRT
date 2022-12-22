#include "facenoise.h"

#include <opencv2/imgproc.hpp>

namespace cv { namespace ofrt {

NoiseDetector::NoiseDetector(const std::string &modelfilename) :
    FaceClassifier(cv::Size(100,100),48.0f,-0.2f)
{
    try {
        dlib::noise::net_type net;
        dlib::deserialize(modelfilename) >> net;
        snet.subnet() = net.subnet();
    } catch(const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
}

std::vector<float> NoiseDetector::process(const cv::Mat &img, const std::vector<cv::Point2f> &landmarks, bool fast)
{
    cv::Mat cv_facepatch = extractFacePatch(img,landmarks,iod(),size(),0,v2hshift(),true,cv::INTER_AREA);
    std::vector<dlib::matrix<dlib::rgb_pixel>> crops(1,cvmat2dlibmatrix(cv_facepatch));
    if(!fast)
        crops.emplace_back(dlib::fliplr(crops[0]));
    dlib::matrix<float,1,2> p = dlib::sum_rows(dlib::mat(snet(crops.begin(),crops.end()))) / crops.size();
    return std::vector<float>(1,p(1));
}

Ptr<FaceClassifier> NoiseDetector::createClassifier(const std::string &modelfilename)
{
    return makePtr<NoiseDetector>(modelfilename);
}

}}