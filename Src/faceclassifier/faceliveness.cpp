#include "faceliveness.h"

#include <opencv2/imgproc.hpp>

namespace cv { namespace ofrt {

FaceLiveness::FaceLiveness(const std::string &modelfilename):
    FaceClassifier(cv::Size(80,80),20.0f,0.0f)
{
    try {
        dlib::liveness_type net;
        dlib::deserialize(modelfilename) >> net;
        snet.subnet() = net.subnet();
    } catch(const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
}

std::vector<float> FaceLiveness::process(const Mat &img, const std::vector<Point2f> &landmarks, bool fast)
{
    cv::Mat cv_facepatch = extractFacePatch(img,landmarks,iod(),size(),0,v2hshift(),true,cv::INTER_LINEAR);
    std::vector<dlib::matrix<dlib::rgb_pixel>> crops(1,cvmat2dlibmatrix(cv_facepatch));
    if(!fast)
        crops.emplace_back(dlib::fliplr(crops[0]));
    dlib::matrix<float,1,4> p = dlib::sum_rows(dlib::mat(snet(crops.begin(),crops.end()))) / crops.size();
    return std::vector<float>(1,p(0));
}

Ptr<FaceClassifier> FaceLiveness::createClassifier(const std::string &modelfilename)
{
    return makePtr<FaceLiveness>(modelfilename);
}

}}
