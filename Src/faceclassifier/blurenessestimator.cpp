#include "blurenessestimator.h"

#include <opencv2/imgproc.hpp>

namespace cv { namespace ofrt {

BlurenessEstimator::BlurenessEstimator(const String &modelfilename) :
    FaceClassifier(cv::Size(100,100),50.0f,-0.1f)
{
    net = cv::dnn::readNet(modelfilename);
    CV_Assert(!net.empty());
#ifdef FORCE_OPENCV_DNN_TO_USE_CUDA
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
#endif
    // Now read names of output layers
    std::vector<int> outLayers = net.getUnconnectedOutLayers();
    std::vector<String> layersNames = net.getLayerNames();
    output_names.resize(outLayers.size());
    for(size_t i = 0; i < outLayers.size(); ++i)
        output_names[i] = layersNames[static_cast<size_t>(outLayers[i]) - 1];
}

std::vector<float> BlurenessEstimator::process(const Mat &img, const std::vector<Point2f> &landmarks, bool fast)
{
    std::vector<cv::Mat> views(1,extractFacePatch(img,landmarks,iod(),size(),0,v2hshift(),true,cv::INTER_LINEAR));
    if(!fast) {
        views.push_back(cv::Mat());
        cv::flip(views[0],views[1],1);
    }
    cv::Mat blob;
    // trained with mean=[0.455, 0.455, 0.455], std=[0.225, 0.225, 0.225]
    cv::dnn::blobFromImages(views,blob,1.0/57.375,cv::Size(),cv::Scalar(116.025,116.025,116.025),false,false);
    std::vector<Mat> output_blobs;
    net.setInput(blob);
    net.forward(output_blobs, output_names);
    float *logits = reinterpret_cast<float*>(output_blobs[0].data);
    const size_t step = output_blobs[0].total() / views.size();
    std::vector<std::vector<float>> bprobs(views.size());
    for(size_t i = 0 ; i < bprobs.size(); ++i)
        bprobs[i] = FaceClassifier::softmax(logits + i*step, step);
    std::vector<float> prob(1,0.0f);
    for(size_t i = 0 ; i < bprobs.size(); ++i)
        prob[0] += bprobs[i][0]; // 0 - label of blureness
    prob[0] /= bprobs.size();
    return prob;
}

Ptr<FaceClassifier> BlurenessEstimator::createClassifier(const String &modelfilename)
{
    return makePtr<BlurenessEstimator>(modelfilename);
}

}}
