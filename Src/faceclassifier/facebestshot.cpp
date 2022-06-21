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

std::vector<float> FaceBestshot::classify(const cv::Mat &img, const std::vector<cv::Point2f> &landmarks)
{
    cv::Mat cv_facepatch = extractFacePatch(img,landmarks,iod(),size(),0,v2hshift(),true,cv::INTER_LINEAR);
    dlib::matrix<dlib::rgb_pixel> dlib_facepatch = cvmat2dlibmatrix(cv_facepatch);
    dlib::matrix<float,1,2> p = dlib::mat(snet(dlib_facepatch));
    std::vector<float> probs(dlib::num_columns(p),0);
    for(size_t i = 0; i < probs.size(); ++i)
        probs[i] = p(i);
    return probs;
}

float FaceBestshot::confidence(const Mat &img, const std::vector<Point2f> &landmarks)
{
    return classify(img,landmarks)[1];
}

Ptr<FaceClassifier> FaceBestshot::createClassifier(const std::string &modelfilename)
{
    return makePtr<FaceBestshot>(modelfilename);
}


}}
