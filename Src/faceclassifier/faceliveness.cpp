#include "faceliveness.h"

#include <opencv2/imgproc.hpp>

namespace cv { namespace ofrt {

FaceLiveness::FaceLiveness(const std::string &modelfilename):
    FaceClassifier(cv::Size(300,300),75.0f,0.0f)
{
    try {
        dlib::liveness_type net;
        dlib::deserialize(modelfilename) >> net;
        snet.subnet() = net.subnet();
    } catch(const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
}

Ptr<FaceClassifier> FaceLiveness::createClassifier(const std::string &modelfilename)
{
    return makePtr<FaceLiveness>(modelfilename);
}

}}
