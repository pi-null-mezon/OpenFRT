#ifndef FACEMARKLITECNN_H
#define FACEMARKLITECNN_H

#include <opencv2/core.hpp>

#include "facemark.h"

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

template <typename SUBNET> using alite2 = ares_down<32,SUBNET>;
template <typename SUBNET> using alite3 = ares<16,ares_down<16,SUBNET>>;
template <typename SUBNET> using alite4 = ares<8,SUBNET>;

using facelandmarks_lite_net_type = loss_mean_squared_multioutput<fc_no_bias<136,avg_pool_everything<
                                                    alite2<
                                                    alite3<
                                                    alite4<
                                                    relu<affine<con<4,3,3,2,2,
                                                    input_rgb_image>>>>>>>>>;
}

namespace cv { namespace ofrt {

/**
 * @brief The FacemarkCNN class is a custom 68 facial points detector based on CNN
 */
class FacemarkLiteCNN : public Facemark {

public:
    FacemarkLiteCNN(const String &model);

    bool fit(const cv::Mat &image,
             const std::vector<Rect> &faces,
             std::vector<std::vector<Point2f>> &landmarks) const override;

    static Ptr<Facemark> create(const String &model);

private:
    mutable dlib::facelandmarks_lite_net_type net;
    cv::Size isize;
};

}}

#endif // FACEMARKLITECNN_H
