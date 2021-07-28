#include "cnnfacedetector.h"

#include <opencv2/imgproc.hpp>

namespace cv { namespace ofrt {

CNNFaceDetector::CNNFaceDetector(const std::string &_txtfilename, const std::string &_modelfilename, float _confidenceThreshold) :
    FaceDetector(),
    confidenceThreshold(_confidenceThreshold)
{
    net = cv::dnn::readNet(_modelfilename,_txtfilename);
#ifdef FORCE_OPENCV_DNN_TO_USE_CUDA
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
#endif
    std::vector<int> outLayers = net.getUnconnectedOutLayers();
    std::vector<String> layersNames = net.getLayerNames();
    outputs.resize(outLayers.size());
    for(size_t i = 0; i < outLayers.size(); ++i) {
        outputs[i] = layersNames[static_cast<size_t>(outLayers[i]) - 1];
    }
}

CNNFaceDetector::~CNNFaceDetector()
{
}

std::vector<Rect> CNNFaceDetector::detectFaces(InputArray &_img) const
{
#ifdef FORCE_OPENCV_DNN_TO_USE_CUDA
    cv::Size _targetsize(131,131);
#else
    cv::Size _targetsize(131,131);
#endif
    float _sX,_sY;
    cv::Point2f _oshift;
    cv::Mat _fixedcanvasimg = resizeAndPasteInCenterOfCanvas(_img.getMat(),_targetsize,_oshift,_sX,_sY);
    cv::Mat blob;
    cv::dnn::blobFromImage(_fixedcanvasimg,blob,1.0,cv::Size(),cv::Scalar(104,177,123),false,false);
    net.setInput(blob);
    std::vector<cv::Mat> outs;
    net.forward(outs,outputs);
    std::vector<Rect> boxes;
    boxes.reserve(32);
    // Network produces output blob with a shape 1x1xNx7 where N is a number of
    // detections and an every detection is a vector of values
    // [batchId, classId, confidence, left, top, right, bottom]
    float* data = reinterpret_cast<float*>(outs[0].data);
    for (size_t i = 0; i < outs[0].total(); i += 7) {
        float confidence = data[i + 2];
        if (confidence > confidenceThreshold) {
            int left   = static_cast<int>((data[i + 3] * _fixedcanvasimg.cols - _oshift.x) / _sX);
            int top    = static_cast<int>((data[i + 4] * _fixedcanvasimg.rows - _oshift.y) / _sY);
            int right  = static_cast<int>((data[i + 5] * _fixedcanvasimg.cols - _oshift.x) / _sX);
            int bottom = static_cast<int>((data[i + 6] * _fixedcanvasimg.rows - _oshift.y) / _sY);
            int width  = right - left + 1;
            int height = bottom - top + 1;
            boxes.push_back( Rect(static_cast<int>(left + width*getXShift()  - width*(getXPortion()  - 1.0f)/2.0f),
                                  static_cast<int>(top  + height*getYShift() - height*(getYPortion() - 1.0f)/2.0f),
                                  static_cast<int>(width*getXPortion()),
                                  static_cast<int>(height*getYPortion())) );
        }
    }
    return boxes;
}

Mat CNNFaceDetector::resizeAndPasteInCenterOfCanvas(const Mat &_img, const Size &_canvassize, cv::Point2f &_originshift, float &_scaleX, float &_scaleY) const
{
    cv::Mat  _canvasmat = cv::Mat::zeros(_canvassize,_img.type());
    cv::Size _targetsize;
    if(static_cast<float>(_canvassize.width)/_canvassize.height > static_cast<float>(_img.cols)/_img.rows) {
        _targetsize.height = _canvassize.height;
        _targetsize.width  = static_cast<int>(_canvassize.height * static_cast<float>(_img.cols)/_img.rows);
    } else {
        _targetsize.width  = _canvassize.width;
        _targetsize.height = static_cast<int>(_canvassize.width * static_cast<float>(_img.rows)/_img.cols);
    }
    _scaleX = static_cast<float>(_targetsize.width)  / _img.cols;
    _scaleY = static_cast<float>(_targetsize.height) / _img.rows;

    int _interptype = cv::INTER_AREA;
    if(_targetsize.area() > (_img.rows*_img.cols)) {
        _interptype = cv::INTER_LINEAR;
    }
    cv::Mat _resizedimg;
    cv::resize(_img,_resizedimg,_targetsize,0,0,_interptype);

    cv::Rect _canvascenterregion = cv::Rect((_canvassize.width - _targetsize.width)/2,
                                            (_canvassize.height - _targetsize.height)/2,
                                             _targetsize.width,
                                             _targetsize.height) & cv::Rect(0,0,_canvassize.width,_canvassize.height);
    _originshift = _canvascenterregion.tl();
    _resizedimg.copyTo(_canvasmat(_canvascenterregion));
    return _canvasmat;
}


Ptr<FaceDetector> CNNFaceDetector::createDetector(const std::string &_txtfilename, const std::string &_modelfilename, float _confidenceThreshold)
{
    return makePtr<CNNFaceDetector>(_txtfilename,_modelfilename,_confidenceThreshold);
}


}}

