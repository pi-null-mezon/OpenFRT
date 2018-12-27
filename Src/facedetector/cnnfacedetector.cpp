#include "cnnfacedetector.h"

namespace cv { namespace ofrt {

CNNFaceDetector::CNNFaceDetector(const std::string &_txtfilename, const std::string &_modelfilename, float _confidenceThreshold) :
    FaceDetector(),
    confidenceThreshold(_confidenceThreshold)
{
    net = cv::dnn::readNet(_modelfilename,_txtfilename);
    std::vector<int> outLayers = net.getUnconnectedOutLayers();
    std::vector<String> layersNames = net.getLayerNames();
    outputs.resize(outLayers.size());
    for(size_t i = 0; i < outLayers.size(); ++i) {
        outputs[i] = layersNames[outLayers[i] - 1];
    }
}

CNNFaceDetector::~CNNFaceDetector()
{
}

std::vector<Rect> CNNFaceDetector::detectFaces(InputArray &_img) const
{
    cv::Mat blob;
    cv::Size _targetsize(96 * _img.rows()/_img.cols(), 96); // greater values could provide detection of smaller faces but processing time will increase proportionally
    cv::dnn::blobFromImage(_img,blob,1,_targetsize,cv::Scalar(104,177,123),false,false);
    net.setInput(blob);
    std::vector<cv::Mat> outs;
    net.forward(outs,outputs);
    std::vector<Rect> boxes;
    boxes.reserve(32);
    // Network produces output blob with a shape 1x1xNx7 where N is a number of
    // detections and an every detection is a vector of values
    // [batchId, classId, confidence, left, top, right, bottom]
    float* data = (float*)outs[0].data;
    for (size_t i = 0; i < outs[0].total(); i += 7) {
        float confidence = data[i + 2];
        if (confidence > confidenceThreshold) {
            int left = (int)(data[i + 3] * _img.cols());
            int top = (int)(data[i + 4] * _img.rows());
            int right = (int)(data[i + 5] * _img.cols());
            int bottom = (int)(data[i + 6] * _img.rows());
            int width = right - left + 1;
            int height = bottom - top + 1;
            boxes.push_back(Rect(left + getXShift()*width - width*(getXPortion() - 1.0f)/2.0f,
                                 top + getYShift()*height - height*(getYPortion() - 1.0f)/2.0f,
                                 width*getXPortion(),
                                 height*getYPortion()));
        }
    }
    return boxes;
}

Ptr<FaceDetector> CNNFaceDetector::createDetector(const std::string &_txtfilename, const std::string &_modelfilename, float _confidenceThreshold)
{
    return makePtr<CNNFaceDetector>(_txtfilename,_modelfilename,_confidenceThreshold);
}

}}

