#ifndef CNNFACEDETECTOR_H
#define CNNFACEDETECTOR_H

#include <opencv2/dnn.hpp>

#include "facedetector.h"

namespace cv { namespace ofrt {

/**
 * @brief The CNNFaceDetector
 * Network description: https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector/deploy.prototxt
 * Network weights:     https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
 */
class CNNFaceDetector: public FaceDetector
{
public:
    CNNFaceDetector(const std::string &_txtfilename, const std::string &_modelfilename, float _confidenceThreshold);
    virtual ~CNNFaceDetector();
    std::vector<cv::Rect> detectFaces(InputArray &_img) const override;
    static Ptr<FaceDetector> createDetector(const std::string &_txtfilename="./deploy_lowres.prototxt",
                                            const std::string &_modelfilename="./res10_300x300_ssd_iter_140000_fp16.caffemodel",
                                            float _confidenceThreshold=0.5f);

private:
    cv::Mat resizeAndPasteInCenterOfCanvas(const cv::Mat &_img, const cv::Size &_canvassize, cv::Point2f &_originshift, float &_scaleX, float &_scaleY) const;
    float confidenceThreshold;
    mutable cv::dnn::Net net;
    std::vector<String> outputs;
};

}}

#endif // CNNFACEDETECTOR_H
