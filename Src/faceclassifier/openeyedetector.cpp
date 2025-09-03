#include "openeyedetector.h"

#include <opencv2/imgproc.hpp>

namespace cv { namespace ofrt {

OpenEyeDetector::OpenEyeDetector(const std::string &modelfilename) :
    FaceClassifier(cv::Size(240,320),80.0f,0)
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

std::vector<float> OpenEyeDetector::process(const Mat &img, const std::vector<Point2f> &landmarks, bool fast)
{
    std::vector<cv::Mat> patches = extractEyesPatches(img,landmarks,iod(),size());
    for(size_t i = 0; i < patches.size(); ++i)
        cv::resize(patches[i],patches[i],cv::Size(40,40),0,0,cv::INTER_AREA);
    if(!fast) {
        for(size_t i = 0; i < 2; ++i) {
            patches.push_back(cv::Mat());
            cv::flip(patches[i],patches[i+2],1);
        }
    }

    cv::Mat blob;
    // trained with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    cv::dnn::blobFromImages(patches,blob,1.0/57.12,cv::Size(),cv::Scalar(123.675, 116.28, 103.53),true,false);
    std::vector<Mat> output_blobs;
    net.setInput(blob);
    net.forward(output_blobs, output_names);
    float *logits = reinterpret_cast<float*>(output_blobs[0].data);
    const size_t step = output_blobs[0].total() / patches.size();
    std::vector<std::vector<float>> bprobs(patches.size()); // will represent openeye prob for [right, left, right_flipped, left_flipped]
    for(size_t i = 0 ; i < bprobs.size(); ++i)
        bprobs[i] = FaceClassifier::softmax(logits + i*step, step);
    std::vector<float> prob(2,0.0f); // [right, left]
    for(size_t i = 0 ; i < bprobs.size(); ++i)
        prob[i % 2] += bprobs[i][1];
    for(size_t i = 0 ; i < prob.size(); ++i)
        prob[i] /= (patches.size() / 2);
    return prob;
}

cv::Ptr<FaceClassifier> OpenEyeDetector::createClassifier(const std::string &modelfilename)
{
    return makePtr<OpenEyeDetector>(modelfilename);
}

std::vector<cv::Mat> OpenEyeDetector::extractEyesPatches(const cv::Mat &_rgbmat, const std::vector<cv::Point2f> &_landmarks, float _targeteyesdistance, const cv::Size &_targetsize)
{
    std::vector<cv::Mat> patches;
    patches.reserve(2);
    cv::Point2f rc(0,0), lc(0,0);
    if(_landmarks.size() == 68) {
        static const uint8_t reye[] = {36,37,38,39,40,41};
        static const uint8_t leye[] = {42,43,44,45,46,47};
        int len = sizeof(reye)/sizeof(reye[0]);
        for(int i = 0; i < len; ++i) {
            rc += _landmarks[reye[i]];
            lc += _landmarks[leye[i]];
        }
        rc /= len;
        lc /= len;
    } else if (_landmarks.size() == 5) {
        rc = _landmarks[0];
        lc = _landmarks[1];
    }

    cv::Point2f cd = lc - rc;
    float _eyesdistance = std::sqrt((cd.x)*(cd.x) + (cd.y)*(cd.y));
    float _scale = _targeteyesdistance / _eyesdistance;
    float _angle = 180.0f * static_cast<float>(std::atan(cd.y/cd.x) / CV_PI);
    cv::Point2f _cp = (rc + lc)/2.0f;
    cv::Mat _tm = cv::getRotationMatrix2D(_cp,_angle,_scale);
    _tm.at<double>(0,2) += _targetsize.width/2.0 - _cp.x;
    _tm.at<double>(1,2) += _targetsize.height/2.0 - _cp.y;

    int _interpolationtype = cv::INTER_LINEAR;
    if(_scale < 1.0f)
        _interpolationtype = cv::INTER_AREA;
    cv::Mat _normalizedfaceimg;
    cv::warpAffine(_rgbmat,_normalizedfaceimg,_tm,_targetsize,_interpolationtype);
    const cv::Rect imgrect(0,0,_normalizedfaceimg.cols,_normalizedfaceimg.rows);
    // We should calculate center points in the new CS
    cv::Point2d nrc(rc.x*_tm.at<double>(0,0) + rc.y*_tm.at<double>(0,1) + _tm.at<double>(0,2),
                     rc.x*_tm.at<double>(1,0) + rc.y*_tm.at<double>(1,1) + _tm.at<double>(1,2));
    cv::Point2d nlc(lc.x*_tm.at<double>(0,0) + lc.y*_tm.at<double>(0,1) + _tm.at<double>(0,2),
                     lc.x*_tm.at<double>(1,0) + lc.y*_tm.at<double>(1,1) + _tm.at<double>(1,2));

    cv::RotatedRect rotatedrect(nrc,cv::Size(_targetsize.width/5,_targetsize.width/5),0);
    cv::Rect roirect = rotatedrect.boundingRect() & imgrect;
    patches.push_back(_normalizedfaceimg(roirect));

    rotatedrect.center = nlc;
    roirect = rotatedrect.boundingRect() & imgrect;
    patches.push_back(_normalizedfaceimg(roirect));
    return patches;
}

}}

