#ifndef RETINAFACEDETECTOR_H
#define RETINAFACEDETECTOR_H

#include <opencv2/dnn.hpp>

#include "facedetector.h"

namespace cv { namespace ofrt {

class RetinaFaceDetector: public FaceDetector
{
public:
    RetinaFaceDetector(int _inputW, int _inputH, const std::string &_modelfilename, float _scoreThreshold);
    virtual ~RetinaFaceDetector();
    std::vector<cv::Rect> detectFaces(InputArray &_img) const override;
    std::vector<std::vector<cv::Point2f>> getLandmarks() const;
    static Ptr<FaceDetector> create(const std::string &_modelfilename="./retina_det_10g.onnx",
                                            int _inputW=160, int _inputH=160, float _confidenceThreshold=0.5f);

private:
    std::vector<Mat> precomputeAnchorCenters() const;
    Mat distance2bbox(const Mat& points, const Mat& distance) const;
    Mat distance2kps(const Mat& points, const Mat& distance) const;
    cv::Mat postProcess(std::vector<Mat> &output_blobs) const;
    cv::Mat resizeAndPasteInCenterOfCanvas(const cv::Mat &_img, const cv::Size &_canvassize, cv::Point2f &_originshift, float &_scaleX, float &_scaleY) const;

    float scoreThreshold;
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
