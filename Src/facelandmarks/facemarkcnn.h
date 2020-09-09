#ifndef FACEMARKCNN_H
#define FACEMARKCNN_H

#include <opencv2/core.hpp>
#include <opencv2/face/facemark.hpp>

#include <dlib/dnn.h>

#define FNUM 8

namespace dlib {

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel1 = ares_down<16*FNUM,SUBNET>;
template <typename SUBNET> using alevel2 = ares<8*FNUM,ares_down<8*FNUM,SUBNET>>;
template <typename SUBNET> using alevel3 = ares<4*FNUM,ares_down<4*FNUM,SUBNET>>;
template <typename SUBNET> using alevel4 = ares<2*FNUM,ares<2*FNUM,SUBNET>>;

using facelandmarks_net_type = loss_mean_squared_multioutput<fc_no_bias<136,avg_pool_everything<
                                                    alevel1<
                                                    alevel2<
                                                    alevel3<
                                                    alevel4<
                                                    relu<affine<con<FNUM,5,5,2,2,
                                                    input_rgb_image >>>>>>>>>>;

}

namespace cv { namespace face {

/**
 * @brief The FacemarkCNN class is a custom 68 facial points detector based on CNN
 */
class FacemarkCNN : public Facemark {

public:
    FacemarkCNN();

    void loadModel(String model);

    bool fit( InputArray image,
              InputArray faces,
              OutputArrayOfArrays landmarks);

private:
    mutable dlib::facelandmarks_net_type net;
    cv::Size isize;
};

Ptr<Facemark> createFacemarkCNN();

}}

#endif // FACEMARKCNN_H
