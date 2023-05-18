#include "facemarkonnx.h"

#include <opencv2/imgproc.hpp>

namespace cv { namespace ofrt {

FacemarkONNX::FacemarkONNX(const String &modelfilename) :
    Facemark(),
    isize(cv::Size(100,100))
{
    net = cv::dnn::readNet(modelfilename);
    CV_Assert(!net.empty());
#ifdef FORCE_OPENCV_DNN_TO_USE_CUDA
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
#endif
}

bool FacemarkONNX::fit(const cv::Mat &image, const std::vector<Rect> &faces, std::vector<std::vector<Point2f>> &landmarks) const
{
    if(image.empty() || (faces.size() < 1))
        return false;    
    landmarks.reserve(faces.size());
    const cv::Rect2f frame(0,0,image.cols,image.rows);
    for(const auto &rect : faces) {
        cv::Rect2f _rect = prepareRect(rect,frame,1.4f);
        cv::Rect2f _roirect;       
        cv::Mat blob;
        // trained with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        cv::dnn::blobFromImage(cropInsideFromCenterAndResize(image(_rect & frame),isize,_roirect),
                               blob,1.0/57.12,cv::Size(),cv::Scalar(123.675, 116.28, 103.53),true,false);
        static std::vector<String> output_names = {"output"};
        std::vector<Mat> output_blobs;
        net.setInput(blob);
        net.forward(output_blobs, output_names);

        float* data = reinterpret_cast<float*>(output_blobs[0].data);
        std::vector<cv::Point2f> _points;
        _points.reserve(output_blobs[0].total() / 2);
        for(size_t i = 0; i < output_blobs[0].total() / 2; ++i)
            _points.push_back(cv::Point2f((0.5f + data[2*i]) * _roirect.width + _rect.x + _roirect.x,
                                          (0.5f + data[2*i+1]) * _roirect.height + _rect.y + _roirect.y));
        landmarks.push_back(std::move(_points));
    }
    return true;
}

Ptr<Facemark> FacemarkONNX::create(const String &modelfilename)
{
    return makePtr<FacemarkONNX>(modelfilename);
}

}}
