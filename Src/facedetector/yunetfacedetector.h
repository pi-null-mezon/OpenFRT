#ifndef YUNETFACEDETECTOR_H
#define YUNETFACEDETECTOR_H

#include <opencv2/dnn.hpp>

#include "facedetector.h"

namespace cv { namespace ofrt {

/**
 * @brief The YuNet Face Detector
 * Network weights:     https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet
 */
class YuNetFaceDetector: public FaceDetector
{
public:
    YuNetFaceDetector(const std::string &_modelfilename, float _scoreThreshold);
    virtual ~YuNetFaceDetector();
    std::vector<cv::Rect> detectFaces(InputArray &_img) const override;
    std::vector<std::vector<cv::Point2f>> detectLandmarks(InputArray &_img) const;
    static Ptr<FaceDetector> createDetector(const std::string &_modelfilename="./face_detection_yunet_2022mar.onnx",
                                            float _confidenceThreshold=0.9f);

private:
    void generatePriors();
    std::vector<cv::Rect2f> priors;

    cv::Mat postProcess(const std::vector<cv::Mat>& output_blobs) const;

    cv::Mat resizeAndPasteInCenterOfCanvas(const cv::Mat &_img, const cv::Size &_canvassize, cv::Point2f &_originshift, float &_scaleX, float &_scaleY) const;

    float scoreThreshold;
    int inputW;
    int inputH;
    float nmsThreshold;
    int topK;
    mutable cv::dnn::Net net;

};

}}

#endif // YUNETFACEDETECTOR_H
