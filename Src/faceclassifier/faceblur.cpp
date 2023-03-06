#include "faceblur.h"

#include <opencv2/imgproc.hpp>

static dlib::matrix<dlib::rgb_pixel> cvmat2dlibmatrix(const cv::Mat &_cvmat)
{
    cv::Mat _mat = _cvmat;
    if(_cvmat.isContinuous() == false)
        _mat = _cvmat.clone();
    unsigned char *_p = _mat.ptr<unsigned char>(0);
    dlib::matrix<dlib::rgb_pixel> _img(_mat.rows,_mat.cols);
    for(long i = 0; i < static_cast<long>(_mat.total()); ++i)
        _img(i) = dlib::rgb_pixel(_p[3*i+2],_p[3*i+1],_p[3*i]); // BGR to RGB
    return _img;
}

namespace cv { namespace ofrt {

FaceBlur::FaceBlur(const std::string &modelfilename) :
    FaceClassifier(cv::Size(100,100),40.0f,-0.2f)
{
    try {
        dlib::deserialize(modelfilename) >> net;
    } catch(const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
}

std::vector<float> FaceBlur::process(const cv::Mat &img, const std::vector<cv::Point2f> &landmarks, bool fast)
{
    cv::Mat cv_facepatch = extractFacePatch(img,landmarks,iod(),size(),0,v2hshift(),true,cv::INTER_LINEAR);
    std::vector<dlib::matrix<dlib::rgb_pixel>> crops(1,cvmat2dlibmatrix(cv_facepatch));
    if(!fast)
        crops.emplace_back(dlib::fliplr(crops[0]));
    dlib::matrix<float,1,1> p = dlib::sum_rows(dlib::mat(net(crops))) / crops.size();
    return std::vector<float>(1,std::min(1.0f,std::max(0.0f,p(0) + 0.5f)));
}

Ptr<FaceClassifier> FaceBlur::createClassifier(const std::string &modelfilename)
{
    return makePtr<FaceBlur>(modelfilename);
}

}}
