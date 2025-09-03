#include "yunet2023fd.h"

#include <opencv2/imgproc.hpp>

namespace cv { namespace ofrt {

YuNetFaceDetector2023::YuNetFaceDetector2023(const std::string &_modelfilename, float _scoreThreshold) :
    FaceDetector(),
    scoreThreshold(_scoreThreshold),
    divisor(32),
    strides({8, 16, 32})
{
    net = cv::dnn::readNet(_modelfilename);
    CV_Assert(!net.empty());
#ifdef FORCE_OPENCV_DNN_TO_USE_CUDA
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
#endif
    nmsThreshold = 0.3f;
    topK = 512;
#ifdef CNN_FACE_DETECTOR_INPUT_SIZE
    inputW = CNN_FACE_DETECTOR_INPUT_SIZE;
    inputH = CNN_FACE_DETECTOR_INPUT_SIZE;
#else
    inputW = 64;
    inputH = 64;
#endif
    padW = (int((inputW - 1) / divisor) + 1) * divisor;
    padH = (int((inputH - 1) / divisor) + 1) * divisor;
    output_names = { "cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32", "bbox_8", "bbox_16", "bbox_32", "kps_8", "kps_16", "kps_32" };
}

YuNetFaceDetector2023::~YuNetFaceDetector2023()
{
}

std::vector<Rect> YuNetFaceDetector2023::detectFaces(InputArray &_img) const
{
    cv::Size _targetsize(inputW,inputH);
    float _sX,_sY;
    cv::Point2f _oshift;
    cv::Mat _fixedcanvasimg = resizeAndPasteInCenterOfCanvas(_img.getMat(),_targetsize,_oshift,_sX,_sY);
    cv::Mat input_blob = dnn::blobFromImage(_fixedcanvasimg);
    // Forward
    std::vector<Mat> output_blobs;
    net.setInput(input_blob);
    net.forward(output_blobs, output_names);

    // Post process
    Mat faces = postProcess(output_blobs);
    std::vector<Rect> boxes;
    boxes.reserve(32);
    for (int i = 0; i < faces.rows; i++) {
        float left   = (faces.at<float>(i, 0) - _oshift.x) / _sX;
        float top    = (faces.at<float>(i, 1) - _oshift.y) / _sY;
        float width  = faces.at<float>(i, 2) / _sX;
        float height = faces.at<float>(i, 3) / _sY;
        boxes.push_back( Rect(static_cast<int>(left + width*getXShift()  - width*(getXPortion()  - 1.0f)/2.0f),
                             static_cast<int>(top  + height*getYShift() - height*(getYPortion() - 1.0f)/2.0f),
                             static_cast<int>(width*getXPortion()),
                             static_cast<int>(height*getYPortion())) );
    }
    return boxes;
}

std::vector<std::vector<Point2f>> YuNetFaceDetector2023::detectLandmarks(InputArray &_img) const
{
    cv::Size _targetsize(inputW,inputH);
    float _sX,_sY;
    cv::Point2f _oshift;
    cv::Mat _fixedcanvasimg = resizeAndPasteInCenterOfCanvas(_img.getMat(),_targetsize,_oshift,_sX,_sY);
    cv::Mat input_blob = dnn::blobFromImage(_fixedcanvasimg);
    // Forward
    std::vector<Mat> output_blobs;
    net.setInput(input_blob);
    net.forward(output_blobs, output_names);
    // Post process
    Mat faces = postProcess(output_blobs);
    std::vector<std::vector<Point2f>> list;
    list.reserve(32);
    for (int i = 0; i < faces.rows; i++) {
        std::vector<Point2f> landmarks(5,cv::Point2f());
        landmarks[0] = cv::Point2f((faces.at<float>(i, 4) - _oshift.x) / _sX, // right eye
                                   (faces.at<float>(i, 5) - _oshift.y) / _sY);
        landmarks[1] = cv::Point2f((faces.at<float>(i, 6) - _oshift.x) / _sX, // left eye
                                   (faces.at<float>(i, 7) - _oshift.y) / _sY);
        landmarks[2] = cv::Point2f((faces.at<float>(i, 8) - _oshift.x) / _sX, // nose tip
                                   (faces.at<float>(i, 9) - _oshift.y) / _sY);
        landmarks[3] = cv::Point2f((faces.at<float>(i, 10) - _oshift.x) / _sX, // right corner of mouth
                                   (faces.at<float>(i, 11) - _oshift.y) / _sY);
        landmarks[4] = cv::Point2f((faces.at<float>(i, 12) - _oshift.x) / _sX, // left corner of mouth
                                   (faces.at<float>(i, 13) - _oshift.y) / _sY);
        list.push_back( std::move(landmarks) );
    }
    return list;
}


Mat YuNetFaceDetector2023::postProcess(const std::vector<Mat>& output_blobs) const
{
    Mat faces;
    for (size_t i = 0; i < strides.size(); ++i) {
        int cols = int(padW / strides[i]);
        int rows = int(padH / strides[i]);

        // Extract from output_blobs
        Mat cls = output_blobs[i];
        Mat obj = output_blobs[i + strides.size() * 1];
        Mat bbox = output_blobs[i + strides.size() * 2];
        Mat kps = output_blobs[i + strides.size() * 3];

        // Decode from predictions
        float* cls_v = (float*)(cls.data);
        float* obj_v = (float*)(obj.data);
        float* bbox_v = (float*)(bbox.data);
        float* kps_v = (float*)(kps.data);

        // (tl_x, tl_y, w, h, re_x, re_y, le_x, le_y, nt_x, nt_y, rcm_x, rcm_y, lcm_x, lcm_y, score)
        // 'tl': top left point of the bounding box
        // 're': right eye, 'le': left eye
        // 'nt':  nose tip
        // 'rcm': right corner of mouth, 'lcm': left corner of mouth
        Mat face(1, 15, CV_32FC1);

        for(int r = 0; r < rows; ++r) {
            for(int c = 0; c < cols; ++c) {
                size_t idx = r * cols + c;

                // Get score
                float cls_score = cls_v[idx];
                float obj_score = obj_v[idx];

                // Clamp
                cls_score = MIN(cls_score, 1.f);
                cls_score = MAX(cls_score, 0.f);
                obj_score = MIN(obj_score, 1.f);
                obj_score = MAX(obj_score, 0.f);
                float score = std::sqrt(cls_score * obj_score);
                face.at<float>(0, 14) = score;

                // Checking if the score meets the threshold before adding the face
                if (score < scoreThreshold)
                    continue;
                // Get bounding box
                float cx = ((c + bbox_v[idx * 4 + 0]) * strides[i]);
                float cy = ((r + bbox_v[idx * 4 + 1]) * strides[i]);
                float w = exp(bbox_v[idx * 4 + 2]) * strides[i];
                float h = exp(bbox_v[idx * 4 + 3]) * strides[i];

                float x1 = cx - w / 2.f;
                float y1 = cy - h / 2.f;

                face.at<float>(0, 0) = x1;
                face.at<float>(0, 1) = y1;
                face.at<float>(0, 2) = w;
                face.at<float>(0, 3) = h;

                // Get landmarks
                for(int n = 0; n < 5; ++n) {
                    face.at<float>(0, 4 + 2 * n) = (kps_v[idx * 10 + 2 * n] + c) * strides[i];
                    face.at<float>(0, 4 + 2 * n + 1) = (kps_v[idx * 10 + 2 * n + 1]+ r) * strides[i];
                }
                faces.push_back(face);
            }
        }
    }

    if (faces.rows > 1)
    {
        // Retrieve boxes and scores
        std::vector<Rect2i> faceBoxes;
        std::vector<float> faceScores;
        for (int rIdx = 0; rIdx < faces.rows; rIdx++)
        {
            faceBoxes.push_back(Rect2i(int(faces.at<float>(rIdx, 0)),
                                       int(faces.at<float>(rIdx, 1)),
                                       int(faces.at<float>(rIdx, 2)),
                                       int(faces.at<float>(rIdx, 3))));
            faceScores.push_back(faces.at<float>(rIdx, 14));
        }

        std::vector<int> keepIdx;
        dnn::NMSBoxes(faceBoxes, faceScores, scoreThreshold, nmsThreshold, keepIdx, 1.f, topK);

        // Get NMS results
        Mat nms_faces;
        for (int idx: keepIdx)
        {
            nms_faces.push_back(faces.row(idx));
        }
        return nms_faces;
    }
    else
    {
        return faces;
    }

}


Mat YuNetFaceDetector2023::resizeAndPasteInCenterOfCanvas(const Mat &_img, const Size &_canvassize, cv::Point2f &_originshift, float &_scaleX, float &_scaleY) const
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

    /*int _interptype = cv::INTER_AREA;
    if(_targetsize.area() > (_img.rows*_img.cols)) {
        _interptype = cv::INTER_LINEAR;
    }*/
    static int _interptype = cv::INTER_LINEAR;
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

Ptr<FaceDetector> YuNetFaceDetector2023::createDetector(const std::string &_modelfilename, float _confidenceThreshold)
{
    return makePtr<YuNetFaceDetector2023>(_modelfilename,_confidenceThreshold);
}


}}

