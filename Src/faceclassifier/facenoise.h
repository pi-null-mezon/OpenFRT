#ifndef FACENOISE_H
#define FACENOISE_H

#include "faceclassifier.h"
#include <dlib/dnn.h>

namespace dlib { namespace noise {

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

using net_type = loss_multiclass_log<fc<2,avg_pool_everything<ares_down<32,ares_down<16,relu<affine<con<8,5,5,2,2,input_rgb_image>>>>>>>>;

}}

namespace cv { namespace ofrt {

class NoiseDetector : public FaceClassifier
{
public:
    NoiseDetector(const std::string &modelfilename);

    std::vector<float> process(const cv::Mat &img, const std::vector<cv::Point2f> &landmarks, bool fast) override;

    static Ptr<FaceClassifier> createClassifier(const std::string &modelfilename="./noise_net.dat");

private:
    dlib::softmax<dlib::noise::net_type::subnet_type> snet;
};

}}

#endif // FACENOISE_H
