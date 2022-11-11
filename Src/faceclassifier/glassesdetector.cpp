#include "glassesdetector.h"

#include <opencv2/imgproc.hpp>

namespace cv { namespace ofrt {

GlassesDetector::GlassesDetector(const std::string &modelfilename) :
    FaceClassifier(cv::Size(100,100),40.0f,0)
{
    try {
        dlib::glasses_net_type net;
        dlib::deserialize(modelfilename) >> net;
        snet.subnet() = net.subnet();
    } catch(const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
}

std::vector<float> GlassesDetector::process(const Mat &img, const std::vector<Point2f> &landmarks, bool fast)
{
    cv::Mat cv_facepatch = extractFacePatch(img,landmarks,iod(),size(),0,v2hshift(),true,cv::INTER_AREA);
    std::vector<dlib::matrix<dlib::rgb_pixel>> crops(1,cvmat2dlibmatrix(cv_facepatch));
    if(!fast)
        crops.emplace_back(dlib::fliplr(crops[0]));
    dlib::matrix<float,1,3> p = dlib::sum_rows(dlib::mat(snet(crops.begin(),crops.end())))/crops.size();
    std::vector<float> probs(3,0.0f);
    probs[0] = p(0); // no glasses
    probs[1] = p(1); // spectacles
    probs[2] = p(2); // sunglasses
    return probs;
}

cv::Ptr<GlassesDetector> GlassesDetector::createClassifier(const std::string &modelfilename)
{
    return makePtr<GlassesDetector>(modelfilename);
}
}}

