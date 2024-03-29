#ifndef FACELIVENESS_H
#define FACELIVENESS_H

#include "faceclassifier.h"
#include <dlib/dnn.h>

namespace dlib {

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

using liveness_type = loss_multiclass_log<fc<4,avg_pool_everything<ares_down<32,ares_down<16,ares<8,relu<affine<con<4,3,3,2,2,input_rgb_image>>>>>>>>>;

}


namespace cv { namespace ofrt {

class FaceLiveness : public FaceClassifier
{
public:
    FaceLiveness(const std::string &modelfilename);

    std::vector<float> process(const cv::Mat &img, const std::vector<cv::Point2f> &landmarks, bool fast) override;

    static Ptr<FaceClassifier> createClassifier(const std::string &modelfilename="./liveness_net_super_lite.dat");

private:
    dlib::softmax<dlib::liveness_type::subnet_type> snet;
};

}}
#endif // FACELIVENESS_H
