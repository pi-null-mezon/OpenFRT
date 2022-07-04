#include "faceblur.h"

#include <opencv2/imgproc.hpp>

namespace cv { namespace ofrt {

FaceBlur::FaceBlur(const std::string &modelfilename) :
    FaceClassifier(cv::Size(150,150),60.0f,-0.2f)
{
    try {
        dlib::deserialize(modelfilename) >> net;
    } catch(const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
}

std::vector<float> FaceBlur::classify(const cv::Mat &img, const std::vector<cv::Point2f> &landmarks)
{
    cv::Mat cv_facepatch = extractFacePatch(img,landmarks,iod(),size(),0,v2hshift(),true,cv::INTER_LINEAR);
    std::vector<dlib::matrix<dlib::rgb_pixel>> crops = make_crops(cv_facepatch);
    dlib::matrix<float,1,1> p = dlib::sum_rows(dlib::mat(net(crops))) / crops.size();
    std::vector<float> blureness(dlib::num_columns(p),0);
    for(size_t i = 0; i < blureness.size(); ++i)
        blureness[i] = std::min(1.0f,std::max(0.0f,p(0) + 0.5f));
    return blureness;
}

float FaceBlur::confidence(const Mat &img, const std::vector<Point2f> &landmarks)
{
    return classify(img,landmarks)[0];
}

Ptr<FaceClassifier> FaceBlur::createClassifier(const std::string &modelfilename)
{
    return makePtr<FaceBlur>(modelfilename);
}

}}
