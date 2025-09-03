#ifndef YUNETFACEDETECTOR2023_H
#define YUNETFACEDETECTOR2023_H

#include <opencv2/dnn.hpp>

#include "facedetector.h"

namespace cv { namespace ofrt {

/**
 * @brief The YuNet Face Detector update
 * Network weights:     https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
 */
class YuNetFaceDetector2023: public FaceDetector
{
public:
    YuNetFaceDetector2023(const std::string &_modelfilename, float _scoreThreshold);
    virtual ~YuNetFaceDetector2023();
    std::vector<cv::Rect> detectFaces(InputArray &_img) const override;
    std::vector<std::vector<cv::Point2f>> detectLandmarks(InputArray &_img) const;
    static Ptr<FaceDetector> createDetector(const std::string &_modelfilename="./face_detection_yunet_2023mar.onnx",
                                            float _confidenceThreshold=0.9f);

private:
    cv::Mat postProcess(const std::vector<cv::Mat>& output_blobs) const;

    cv::Mat resizeAndPasteInCenterOfCanvas(const cv::Mat &_img, const cv::Size &_canvassize, cv::Point2f &_originshift, float &_scaleX, float &_scaleY) const;

    float scoreThreshold;
    int inputW;
    int inputH;
    int divisor, padW, padH;
    const std::vector<int> strides;
    float nmsThreshold;
    int topK;
    std::vector<String> output_names;
    mutable cv::dnn::Net net;
};

}}

#endif // YUNETFACEDETECTOR2023_H
