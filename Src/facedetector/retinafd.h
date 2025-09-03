#ifndef RETINAFACEDETECTOR_H
#define RETINAFACEDETECTOR_H

#include <opencv2/dnn.hpp>

#include "facedetector.h"

namespace cv { namespace ofrt {

/**
 * @brief The YuNet Face Detector update
 * Network weights:     https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
 */
class RetinaFaceDetector: public FaceDetector
{
public:
    RetinaFaceDetector(const std::string &_modelfilename, float _scoreThreshold);
    virtual ~RetinaFaceDetector();
    std::vector<cv::Rect> detectFaces(InputArray &_img) const override;
    std::vector<std::vector<cv::Point2f>> getLandmarks() const;
    static Ptr<FaceDetector> createDetector(const std::string &_modelfilename="./retina_det_10g.onnx", float _confidenceThreshold=0.5f);

private:
    std::vector<Mat> precomputeAnchorCenters() const;
    Mat distance2bbox(const Mat& points, const Mat& distance) const;
    Mat distance2kps(const Mat& points, const Mat& distance) const;
    cv::Mat postProcess(std::vector<Mat> &output_blobs) const;
    cv::Mat resizeAndPasteInCenterOfCanvas(const cv::Mat &_img, const cv::Size &_canvassize, cv::Point2f &_originshift, float &_scaleX, float &_scaleY) const;

    float scoreThreshold;
    int inputW;
    int inputH;
    int fmc;
    const std::vector<int> feat_stride_fpn;
    int num_anchors;
    float nmsThreshold;
    int topK;
    std::vector<String> output_names;
    std::vector<cv::Mat> anchorCenters;
    mutable std::vector<std::vector<cv::Point2f>> list;
    mutable cv::dnn::Net net;
};

}}

#endif // RETINAFACEDETECTOR_H
