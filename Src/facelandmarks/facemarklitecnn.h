#ifndef FACEMARKLITECNN_H
#define FACEMARKLITECNN_H

#include <opencv2/core.hpp>
#include <opencv2/face/facemark.hpp>

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

template <typename SUBNET> using alevel2 = ares<32,ares_down<32,SUBNET>>;
template <typename SUBNET> using alevel3 = ares<16,ares_down<16,SUBNET>>;
template <typename SUBNET> using alevel4 = ares<8,ares_down<8,SUBNET>>;

using facelandmarks_net_type = loss_mean_squared_multioutput<fc_no_bias<136,avg_pool_everything<
                                                    alevel2<
                                                    alevel3<
                                                    alevel4<
                                                    relu<affine<con<4,5,5,2,2,
                                                    input_rgb_image >>>>>>>>>;

}

namespace cv { namespace face {

/**
 * @brief The FacemarkCNN class is a custom 68 facial points detector based on CNN
 */
class FacemarkLiteCNN : public Facemark {

public:
    FacemarkLiteCNN();

    void loadModel(String model);

    bool fit( InputArray image,
              InputArray faces,
              OutputArrayOfArrays landmarks);

private:
    mutable dlib::facelandmarks_net_type net;
    cv::Size isize;
};

Ptr<Facemark> createFacemarkLiteCNN();

}}

#endif // FACEMARKLITECNN_H
