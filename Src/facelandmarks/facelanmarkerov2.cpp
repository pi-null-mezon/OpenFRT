#include "facelanmarkerov2.h"

#include <opencv2/imgproc.hpp>

namespace cv { namespace ofrt {

FaceLandmarkerOV2::FaceLandmarkerOV2(const String &modelfilename) :
    Facemark(),
    isize(cv::Size(48,48))
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

bool FaceLandmarkerOV2::fit(const Mat &image, const std::vector<Rect> &faces, std::vector<std::vector<Point2f>> &landmarks) const
{   
    if(image.empty() || (faces.size() < 1))
        return false;
    landmarks.reserve(faces.size());
    const cv::Rect frame(0,0,image.cols,image.rows);
    confidences = std::vector<std::vector<float>>();
    for(const auto &rect : faces) {
        int original_width = rect.width;
        int original_height = rect.height;
        cv::Mat roimat = image(rect & frame);
        cv::resize(roimat,roimat,isize);
        cv::Mat blob;
        cv::dnn::blobFromImage(roimat,  blob,1.0/128.0,cv::Size(),cv::Scalar(127.5, 127.5, 127.5),false,false);
        std::vector<Mat> output_blobs;
        net.setInput(blob);
        net.forward(output_blobs, output_names);
        // output_blobs[0] - 10 float coordinates of 5 facial landmarks
        // output_blobs[1] - 5 float confidences

        float* data = reinterpret_cast<float*>(output_blobs[0].data);
        std::vector<cv::Point2f> _points;
        _points.reserve(output_blobs[0].total() / 2);
        for(size_t i = 0; i < output_blobs[0].total() / 2; ++i)
            _points.push_back(cv::Point2f(data[i] * original_width + rect.x, data[5 +i] * original_height + rect.y));
        landmarks.push_back(std::move(_points));

        data = reinterpret_cast<float*>(output_blobs[1].data);
        std::vector _conf(5,0.0f);
        for(size_t i = 0; i < output_blobs[1].total(); ++i)
            _conf[i] = data[i];
        confidences.push_back(std::move(_conf));
    }
    return true;
}

std::vector<std::vector<float>> FaceLandmarkerOV2::getConfidences() const
{
    return confidences;
}

Ptr<Facemark> FaceLandmarkerOV2::create(const String &modelfilename)
{
    return makePtr<FaceLandmarkerOV2>(modelfilename);
}

}}
