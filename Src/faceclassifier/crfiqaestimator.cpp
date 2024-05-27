#include "crfiqaestimator.h"

#include <opencv2/imgproc.hpp>

namespace cv { namespace ofrt {

CRFIQAEstimator::CRFIQAEstimator(const String &modelfilename) :
    FaceClassifier(cv::Size(96,112),37.0f,-0.025f)
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

std::vector<float> CRFIQAEstimator::process(const Mat &img, const std::vector<Point2f> &landmarks, bool fast)
{
    std::vector<cv::Mat> views(1,extractFacePatch(img,landmarks,iod(),size(),0,v2hshift(),true,cv::INTER_AREA));
    if(!fast) {
        views.push_back(cv::Mat());
        cv::flip(views[0],views[1],1);
    }
    cv::Mat blob;
    // trained with mean=3*[127.5/255], std=3*[128/255] 
    cv::dnn::blobFromImages(views,blob,1.0/128.0,cv::Size(),cv::Scalar(127.5, 127.5, 127.5),true,false);
    std::vector<Mat> output_blobs;
    net.setInput(blob);
    net.forward(output_blobs, output_names);
    float *logits = reinterpret_cast<float*>(output_blobs[0].data);
    std::vector<float> score(1,0.0f);
    for(size_t i = 0 ; i < output_blobs[0].total(); ++i)
        score[0] += logits[i]; // crfiqa score
    score[0] /= output_blobs[0].total();
    score[0] = std::min(std::max(2.5f * score[0] - 0.75f,0.0f),1.0f);
    return score;
}

Ptr<FaceClassifier> CRFIQAEstimator::createClassifier(const String &modelfilename)
{
    return makePtr<CRFIQAEstimator>(modelfilename);
}


}}
