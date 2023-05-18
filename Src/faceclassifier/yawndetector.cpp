#include "yawndetector.h"

#include <opencv2/imgproc.hpp>

namespace cv { namespace ofrt {

YawnDetector::YawnDetector(const String &modelfilename) :
    FaceClassifier(cv::Size(64,64),25.6f,-0.3f)
{
    net = cv::dnn::readNet(modelfilename);
    CV_Assert(!net.empty());
#ifdef FORCE_OPENCV_DNN_TO_USE_CUDA
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
#endif
    // Now read names of outbut layers
    std::vector<int> outLayers = net.getUnconnectedOutLayers();
    std::vector<String> layersNames = net.getLayerNames();
    output_names.resize(outLayers.size());
    for(size_t i = 0; i < outLayers.size(); ++i)
        output_names[i] = layersNames[static_cast<size_t>(outLayers[i]) - 1];
}

std::vector<float> YawnDetector::process(const Mat &img, const std::vector<Point2f> &landmarks, bool fast)
{
    std::vector<cv::Mat> views(1,extractFacePatch(img,landmarks,iod(),size(),0,v2hshift(),true,cv::INTER_AREA));
    if(!fast) {
        views.push_back(cv::Mat());
        cv::flip(views[0],views[1],1);
    }
    cv::Mat blob;
    // trained with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    cv::dnn::blobFromImages(views,blob,1.0/57.12,cv::Size(),cv::Scalar(123.675, 116.28, 103.53),true,false);
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
        prob[0] += bprobs[i][1]; // 1 - mouth open
    prob[0] /= bprobs.size();
    return prob;
}

Ptr<FaceClassifier> YawnDetector::createClassifier(const String &modelfilename)
{
    return makePtr<YawnDetector>(modelfilename);
}


}}
