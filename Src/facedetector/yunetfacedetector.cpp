#include "yunetfacedetector.h"

#include <opencv2/imgproc.hpp>

namespace cv { namespace ofrt {

YuNetFaceDetector::YuNetFaceDetector(const std::string &_modelfilename, float _scoreThreshold) :
    FaceDetector(),
    scoreThreshold(_scoreThreshold)
{
    net = cv::dnn::readNet(_modelfilename);
    CV_Assert(!net.empty());
#ifdef FORCE_OPENCV_DNN_TO_USE_CUDA
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
#endif
    nmsThreshold = 0.3f;
    topK = 5000;
#ifdef CNN_FACE_DETECTOR_INPUT_SIZE
    inputW = CNN_FACE_DETECTOR_INPUT_SIZE;
    inputH = CNN_FACE_DETECTOR_INPUT_SIZE;
#else
    inputW = 70;
    inputH = 70;
#endif
    generatePriors();
}

YuNetFaceDetector::~YuNetFaceDetector()
{
}

std::vector<Rect> YuNetFaceDetector::detectFaces(InputArray &_img) const
{
    cv::Size _targetsize(inputW,inputH);
    float _sX,_sY;
    cv::Point2f _oshift;
    cv::Mat _fixedcanvasimg = resizeAndPasteInCenterOfCanvas(_img.getMat(),_targetsize,_oshift,_sX,_sY);
    cv::Mat input_blob = dnn::blobFromImage(_fixedcanvasimg);
    std::vector<String> output_names = { "loc", "conf", "iou" };
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

std::vector<std::vector<Point2f>> YuNetFaceDetector::detectLandmarks(InputArray &_img) const
{
    cv::Size _targetsize(inputW,inputH);
    float _sX,_sY;
    cv::Point2f _oshift;
    cv::Mat _fixedcanvasimg = resizeAndPasteInCenterOfCanvas(_img.getMat(),_targetsize,_oshift,_sX,_sY);
    cv::Mat input_blob = dnn::blobFromImage(_fixedcanvasimg);
    std::vector<String> output_names = { "loc", "conf", "iou" };
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

void YuNetFaceDetector::generatePriors()
{
    // Calculate shapes of different scales according to the shape of input image
    Size feature_map_2nd = {
        int(int((inputW+1)/2)/2), int(int((inputH+1)/2)/2)
    };
    Size feature_map_3rd = {
        int(feature_map_2nd.width/2), int(feature_map_2nd.height/2)
    };
    Size feature_map_4th = {
        int(feature_map_3rd.width/2), int(feature_map_3rd.height/2)
    };
    Size feature_map_5th = {
        int(feature_map_4th.width/2), int(feature_map_4th.height/2)
    };
    Size feature_map_6th = {
        int(feature_map_5th.width/2), int(feature_map_5th.height/2)
    };

    std::vector<Size> feature_map_sizes;
    feature_map_sizes.push_back(feature_map_3rd);
    feature_map_sizes.push_back(feature_map_4th);
    feature_map_sizes.push_back(feature_map_5th);
    feature_map_sizes.push_back(feature_map_6th);

    // Fixed params for generating priors
    const std::vector<std::vector<float>> min_sizes = {
        {10.0f,  16.0f,  24.0f},
        {32.0f,  48.0f},
        {64.0f,  96.0f},
        {128.0f, 192.0f, 256.0f}
    };
    CV_Assert(min_sizes.size() == feature_map_sizes.size()); // just to keep vectors in sync
    const std::vector<int> steps = { 8, 16, 32, 64 };

    // Generate priors
    priors.clear();
    for (size_t i = 0; i < feature_map_sizes.size(); ++i)
    {
        Size feature_map_size = feature_map_sizes[i];
        std::vector<float> min_size = min_sizes[i];

        for (int _h = 0; _h < feature_map_size.height; ++_h)
        {
            for (int _w = 0; _w < feature_map_size.width; ++_w)
            {
                for (size_t j = 0; j < min_size.size(); ++j)
                {
                    float s_kx = min_size[j] / inputW;
                    float s_ky = min_size[j] / inputH;

                    float cx = (_w + 0.5f) * steps[i] / inputW;
                    float cy = (_h + 0.5f) * steps[i] / inputH;

                    Rect2f prior = { cx, cy, s_kx, s_ky };
                    priors.push_back(prior);
                }
            }
        }
    }
}

Mat YuNetFaceDetector::postProcess(const std::vector<Mat>& output_blobs) const
{
    // Extract from output_blobs
    Mat loc = output_blobs[0];
    Mat conf = output_blobs[1];
    Mat iou = output_blobs[2];

    // Decode from deltas and priors
    const std::vector<float> variance = {0.1f, 0.2f};
    float* loc_v = (float*)(loc.data);
    float* conf_v = (float*)(conf.data);
    float* iou_v = (float*)(iou.data);
    Mat faces;
    // (tl_x, tl_y, w, h, re_x, re_y, le_x, le_y, nt_x, nt_y, rcm_x, rcm_y, lcm_x, lcm_y, score)
    // 'tl': top left point of the bounding box
    // 're': right eye, 'le': left eye
    // 'nt':  nose tip
    // 'rcm': right corner of mouth, 'lcm': left corner of mouth
    Mat face(1, 15, CV_32FC1);
    for (size_t i = 0; i < priors.size(); ++i) {
        // Get score
        float clsScore = conf_v[i*2+1];
        float iouScore = iou_v[i];
        // Clamp
        if (iouScore < 0.f) {
            iouScore = 0.f;
        }
        else if (iouScore > 1.f) {
            iouScore = 1.f;
        }
        float score = std::sqrt(clsScore * iouScore);
        face.at<float>(0, 14) = score;

        // Get bounding box
        float cx = (priors[i].x + loc_v[i*14+0] * variance[0] * priors[i].width)  * inputW;
        float cy = (priors[i].y + loc_v[i*14+1] * variance[0] * priors[i].height) * inputH;
        float w  = priors[i].width  * exp(loc_v[i*14+2] * variance[0]) * inputW;
        float h  = priors[i].height * exp(loc_v[i*14+3] * variance[1]) * inputH;
        float x1 = cx - w / 2;
        float y1 = cy - h / 2;
        face.at<float>(0, 0) = x1;
        face.at<float>(0, 1) = y1;
        face.at<float>(0, 2) = w;
        face.at<float>(0, 3) = h;

        // Get landmarks
        face.at<float>(0, 4) = (priors[i].x + loc_v[i*14+ 4] * variance[0] * priors[i].width)  * inputW;  // right eye, x
        face.at<float>(0, 5) = (priors[i].y + loc_v[i*14+ 5] * variance[0] * priors[i].height) * inputH;  // right eye, y
        face.at<float>(0, 6) = (priors[i].x + loc_v[i*14+ 6] * variance[0] * priors[i].width)  * inputW;  // left eye, x
        face.at<float>(0, 7) = (priors[i].y + loc_v[i*14+ 7] * variance[0] * priors[i].height) * inputH;  // left eye, y
        face.at<float>(0, 8) = (priors[i].x + loc_v[i*14+ 8] * variance[0] * priors[i].width)  * inputW;  // nose tip, x
        face.at<float>(0, 9) = (priors[i].y + loc_v[i*14+ 9] * variance[0] * priors[i].height) * inputH;  // nose tip, y
        face.at<float>(0, 10) = (priors[i].x + loc_v[i*14+10] * variance[0] * priors[i].width)  * inputW; // right corner of mouth, x
        face.at<float>(0, 11) = (priors[i].y + loc_v[i*14+11] * variance[0] * priors[i].height) * inputH; // right corner of mouth, y
        face.at<float>(0, 12) = (priors[i].x + loc_v[i*14+12] * variance[0] * priors[i].width)  * inputW; // left corner of mouth, x
        face.at<float>(0, 13) = (priors[i].y + loc_v[i*14+13] * variance[0] * priors[i].height) * inputH; // left corner of mouth, y

        faces.push_back(face);
    }

    if (faces.rows > 1) {
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
    else {
        return faces;
    }
}


Mat YuNetFaceDetector::resizeAndPasteInCenterOfCanvas(const Mat &_img, const Size &_canvassize, cv::Point2f &_originshift, float &_scaleX, float &_scaleY) const
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




Ptr<FaceDetector> YuNetFaceDetector::createDetector(const std::string &_modelfilename, float _confidenceThreshold)
{
    return makePtr<YuNetFaceDetector>(_modelfilename,_confidenceThreshold);
}


}}

