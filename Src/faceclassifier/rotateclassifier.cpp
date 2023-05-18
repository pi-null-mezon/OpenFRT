#include "rotateclassifier.h"

#include <opencv2/imgproc.hpp>

#include <opencv2/highgui.hpp>
#include <iostream>


namespace cv { namespace ofrt {

RotateClassifier::RotateClassifier(const String &modelfilename) :
    FaceClassifier(cv::Size(48,48),20.0f,-0.2f)
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

std::vector<float> RotateClassifier::process(const Mat &img, const std::vector<Point2f> &landmarks, bool fast)
{
    return std::vector<float>();
}

std::vector<float> RotateClassifier::process(const Mat &img, const Rect facerect, bool fast)
{
    cv::Mat facepatch = extractFacePatch(img,facerect,size(),1.0f,cv::INTER_LINEAR);
    cv::imshow("rotate", facepatch);
    std::cout << facepatch.total() << std::endl;
    cv::Mat blob;
    cv::dnn::blobFromImage(facepatch,blob,1.0/57.375,cv::Size(),cv::Scalar(116.025,116.025,116.025),false,false);
    std::vector<Mat> output_blobs;
    net.setInput(blob);
    net.forward(output_blobs, output_names);
    float *logits = reinterpret_cast<float*>(output_blobs[0].data);
    const int length = output_blobs[0].total();
    return FaceClassifier::softmax(logits, length);
}

Ptr<FaceClassifier> RotateClassifier::createClassifier(const String &modelfilename)
{
    return makePtr<RotateClassifier>(modelfilename);
}

}}
