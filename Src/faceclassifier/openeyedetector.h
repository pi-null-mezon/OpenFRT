#ifndef OPENEYEDETECTOR_H
#define OPENEYEDETECTOR_H

#include "faceclassifier.h"

namespace dlib {

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using al2 = ares_down<64,SUBNET>;
template <typename SUBNET> using al3 = ares_down<32,SUBNET>;
template <typename SUBNET> using al4 = ares_down<16,SUBNET>;

using blink_net_type = loss_multiclass_log<fc_no_bias<2,avg_pool_everything<al2<al3<al4<relu<affine<con<8,5,5,2,2,input_rgb_image>>>>>>>>>;
}

namespace cv { namespace ofrt {

class OpenEyeDetector : public FaceClassifier
{
public:
    OpenEyeDetector(const std::string &modelfilename);

    std::vector<float> process(const cv::Mat &img, const std::vector<cv::Point2f> &landmarks, bool fast) override;

    static cv::Ptr<FaceClassifier> createClassifier(const std::string &modelfilename="./openeye_net.dat");

    static std::vector<cv::Mat> extractEyesPatches(const cv::Mat &_rgbmat, const std::vector<cv::Point2f> &_landmarks, float _targeteyesdistance, const cv::Size &_targetsize);

private:   
    dlib::softmax<dlib::blink_net_type::subnet_type> snet;
};

}}

#endif // GLASSESDETECTOR_H
