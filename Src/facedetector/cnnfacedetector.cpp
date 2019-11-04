#include "cnnfacedetector.h"

#include <opencv2/imgproc.hpp>

namespace cv { namespace ofrt {

CNNFaceDetector::CNNFaceDetector(const std::string &_txtfilename, const std::string &_modelfilename, float _confidenceThreshold) :
    FaceDetector(),
    confidenceThreshold(_confidenceThreshold)
{
    net = cv::dnn::readNet(_modelfilename,_txtfilename);
#ifdef TRY_TO_USE_CUDA // Tested successfully on Nvidia Jetson Nano with opencv412
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
#endif
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
#ifdef TRY_TO_USE_CUDA
    cv::Size _targetsize(300,300);
#else
    cv::Size _targetsize(180,180);
#endif
    float _sX,_sY;
    cv::Point2f _oshift;
    cv::Mat _fixedcanvasimg = resizeAndPasteInCenterOfCanvas(_img.getMat(),_targetsize,_oshift,_sX,_sY);
    cv::Mat blob;
    cv::dnn::blobFromImage(_fixedcanvasimg,blob,1,cv::Size(),cv::Scalar(104,177,123),false,false);
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
            int left   = (int)((data[i + 3] * _fixedcanvasimg.cols - _oshift.x) / _sX);
            int top    = (int)((data[i + 4] * _fixedcanvasimg.rows - _oshift.y) / _sY);
            int right  = (int)((data[i + 5] * _fixedcanvasimg.cols - _oshift.x) / _sX);
            int bottom = (int)((data[i + 6] * _fixedcanvasimg.rows - _oshift.y) / _sY);
            int width  = right - left + 1;
            int height = bottom - top + 1;
            boxes.push_back( Rect(left + width*getXShift()  - width*(getXPortion()  - 1.0f)/2.0f,
                                  top  + height*getYShift() - height*(getYPortion() - 1.0f)/2.0f,
                                  width*getXPortion(),
                                  height*getYPortion()) );
        }
    }
    return boxes;
}

Mat CNNFaceDetector::resizeAndPasteInCenterOfCanvas(const Mat &_img, const Size &_canvassize, cv::Point2f &_originshift, float &_scaleX, float &_scaleY) const
{
    cv::Mat  _canvasmat = cv::Mat::zeros(_canvassize,_img.type());
    cv::Size _targetsize;
    if((float)_canvassize.width/_canvassize.height > (float)_img.cols/_img.rows) {
        _targetsize.height = _canvassize.height;
        _targetsize.width  = _canvassize.height * (float)_img.cols/_img.rows;
    } else {
        _targetsize.width  = _canvassize.width;
        _targetsize.height = _canvassize.width * (float)_img.rows/_img.cols;
    }
    _scaleX = (float)_targetsize.width  / _img.cols;
    _scaleY = (float)_targetsize.height / _img.rows;

    int _interptype = cv::INTER_AREA;
    if(_targetsize.area() > (_img.rows*_img.cols)) {
        _interptype = cv::INTER_LINEAR;
    }
    cv::Mat _resizedimg;
    cv::resize(_img,_resizedimg,_targetsize,0,0,_interptype);

    cv::Rect _canvascenterregion = cv::Rect2f((_canvassize.width - _targetsize.width)/2.0f,
                                              (_canvassize.height - _targetsize.height)/2.0f,
                                               _targetsize.width,
                                               _targetsize.height) & cv::Rect2f(0,0,_canvassize.width,_canvassize.height);
    _originshift = _canvascenterregion.tl();
    _resizedimg.copyTo(_canvasmat(_canvascenterregion));
    return _canvasmat;
}


Ptr<FaceDetector> CNNFaceDetector::createDetector(const std::string &_txtfilename, const std::string &_modelfilename, float _confidenceThreshold)
{
    return makePtr<CNNFaceDetector>(_txtfilename,_modelfilename,_confidenceThreshold);
}


}}

