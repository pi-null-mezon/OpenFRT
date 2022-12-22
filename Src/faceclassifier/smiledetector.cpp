#include "smiledetector.h"

#include <opencv2/imgproc.hpp>

namespace cv { namespace ofrt {

SmileDetector::SmileDetector(const std::string &modelfilename) :
    FaceClassifier(cv::Size(64,64),30.0f,-0.4f)
{
    try {
        dlib::smile::net_type net;
        dlib::deserialize(modelfilename) >> net;
        snet.subnet() = net.subnet();
    } catch(const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
}

std::vector<float> SmileDetector::process(const cv::Mat &img, const std::vector<cv::Point2f> &landmarks, bool fast)
{
    cv::Mat cv_facepatch = extractFacePatch(img,landmarks,iod(),size(),0,v2hshift(),true,cv::INTER_AREA);
    std::vector<dlib::matrix<dlib::rgb_pixel>> crops(1,cvmat2dlibmatrix(cv_facepatch));
    if(!fast)
        crops.emplace_back(dlib::fliplr(crops[0]));
    dlib::matrix<float,1,2> p = dlib::sum_rows(dlib::mat(snet(crops.begin(),crops.end()))) / crops.size();
    return std::vector<float>(1,p(1));
}

Ptr<FaceClassifier> SmileDetector::createClassifier(const std::string &modelfilename)
{
    return makePtr<SmileDetector>(modelfilename);
}

}}