#ifndef HEADPOSEPREDICTOR_H
#define HEADPOSEPREDICTOR_H

#include "faceclassifier.h"

namespace dlib { namespace headpose {

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using l1 = ares_down<128,SUBNET>;
template <typename SUBNET> using l2 = ares_down<64,SUBNET>;
template <typename SUBNET> using l3 = ares_down<32,SUBNET>;
template <typename SUBNET> using l4 = ares_down<16,SUBNET>;

using net_type = loss_mean_squared_multioutput<fc<3,avg_pool_everything<l1<l2<l3<l4<relu<affine<con<8,5,5,2,2,input_rgb_image>>>>>>>>>>;

}}

namespace cv { namespace ofrt {

class HeadPosePredictor : public FaceClassifier
{
public:
    HeadPosePredictor(const std::string &modelfilename);

    std::vector<float> process(const cv::Mat &img, const std::vector<cv::Point2f> &landmarks, bool fast) override;

    static Ptr<FaceClassifier> createClassifier(const std::string &modelfilename="./headpose_net.dat");

private:
    cv::Mat crop(const Mat &img, const std::vector<Point2f> &landmarks, const cv::Size targetsize);

    dlib::headpose::net_type net;
};

}}

#endif // HEADPOSEPREDICTOR_H
