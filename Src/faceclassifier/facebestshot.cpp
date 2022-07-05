#include "facebestshot.h"

#include <opencv2/imgproc.hpp>

namespace cv { namespace ofrt {

FaceBestshot::FaceBestshot(const std::string &modelfilename) :
    FaceClassifier(cv::Size(80,80),30.0f,-0.1f)
{
    try {
        dlib::bestshot_net_type net;
        dlib::deserialize(modelfilename) >> net;
        snet.subnet() = net.subnet();
    } catch(const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
}

std::vector<float> FaceBestshot::process(const cv::Mat &img, const std::vector<cv::Point2f> &landmarks, bool fast)
{
    cv::Mat cv_facepatch = extractFacePatch(img,landmarks,iod(),size(),0,v2hshift(),true,cv::INTER_LINEAR);
    std::vector<dlib::matrix<dlib::rgb_pixel>> crops(1,cvmat2dlibmatrix(cv_facepatch));
    if(!fast)
        crops.push_back(dlib::fliplr(crops[0]));
    dlib::matrix<float,1,2> p = dlib::sum_rows(dlib::mat(snet(crops.begin(),crops.end()))) / crops.size();
    std::vector<float> probs(dlib::num_columns(p),0);
    for(size_t i = 0; i < probs.size(); ++i)
        probs[i] = p(i);
    return probs;
}

Ptr<FaceClassifier> FaceBestshot::createClassifier(const std::string &modelfilename)
{
    return makePtr<FaceBestshot>(modelfilename);
}

}}
