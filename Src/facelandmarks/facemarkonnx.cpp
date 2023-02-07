#include "facemarkonnx.h"

#include <opencv2/imgproc.hpp>

static cv::Mat cropInsideFromCenterAndResize(const cv::Mat &input, const cv::Size &size, cv::Rect2f &roiRect)
{
    roiRect = cv::Rect2f(0,0,0,0);
    if(static_cast<float>(input.cols)/input.rows > static_cast<float>(size.width)/size.height) {
        roiRect.height = static_cast<float>(input.rows);
        roiRect.width = input.rows * static_cast<float>(size.width)/size.height;
        roiRect.x = (input.cols - roiRect.width)/2.0f;
    } else {
        roiRect.width = static_cast<float>(input.cols);
        roiRect.height = input.cols * static_cast<float>(size.height)/size.width;
        roiRect.y = (input.rows - roiRect.height)/2.0f;
    }
    roiRect &= cv::Rect2f(0.0f, 0.0f, static_cast<float>(input.cols), static_cast<float>(input.rows));
    cv::Mat output;
    if(roiRect.area() > 0)  {
        cv::Mat croppedImg(input, roiRect);
        int interpolationMethod = cv::INTER_AREA;
        if(size.area() > roiRect.area())
            interpolationMethod = cv::INTER_CUBIC;
        cv::resize(croppedImg, output, size, 0, 0, interpolationMethod);
    }
    return output;
}

namespace cv { namespace ofrt {

FacemarkONNX::FacemarkONNX(const String &modelfilename) :
    Facemark(),
    isize(cv::Size(100,100))
{
    net = cv::dnn::readNet(modelfilename);
    CV_Assert(!net.empty());
}

bool FacemarkONNX::fit(const cv::Mat &image, const std::vector<Rect> &faces, std::vector<std::vector<Point2f>> &landmarks) const
{
    if(image.empty() || (faces.size() < 1))
        return false;    
    landmarks.reserve(faces.size());
    const cv::Rect frame(0,0,image.cols,image.rows);
    for(const auto &_rect : faces) {
        cv::Rect2f _roirect;       
        cv::Mat blob;
        cv::dnn::blobFromImage(cropInsideFromCenterAndResize(image(_rect & frame),isize,_roirect),
                               blob,1.0/255.0,cv::Size(),cv::Scalar(0.485, 0.456, 0.406),true,false);

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
