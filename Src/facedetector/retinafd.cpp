#include "retinafd.h"

#include <opencv2/imgproc.hpp>

namespace cv { namespace ofrt {

RetinaFaceDetector::RetinaFaceDetector(int _inputW, int _inputH, const std::string &_modelfilename, float _scoreThreshold) :
    FaceDetector(_inputW,_inputH),
    scoreThreshold(_scoreThreshold),
    fmc(3),
    feat_stride_fpn({8, 16, 32}),
    num_anchors(2),
    nmsThreshold(0.4f),
    topK(512)
{
    net = cv::dnn::readNet(_modelfilename);
    CV_Assert(!net.empty());
#ifdef FORCE_OPENCV_DNN_TO_USE_CUDA
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
#endif
#ifdef CNN_FACE_DETECTOR_INPUT_SIZE
    inputW = CNN_FACE_DETECTOR_INPUT_SIZE;
    inputH = CNN_FACE_DETECTOR_INPUT_SIZE;
#endif
    output_names = {"448","471","494","451","474","497","454","477","500"};
    anchorCenters = precomputeAnchorCenters();
}

RetinaFaceDetector::~RetinaFaceDetector()
{
}

std::vector<Rect> RetinaFaceDetector::detectFaces(InputArray &_img) const
{
    cv::Size _targetsize(inputW(),inputH());
    float _sX,_sY;
    cv::Point2f _oshift;
    cv::Mat _fixedcanvasimg = resizeAndPasteInCenterOfCanvas(_img.getMat(),_targetsize,_oshift,_sX,_sY);
    cv::Mat input_blob = dnn::blobFromImage(_fixedcanvasimg,1.0/128.0,cv::Size(),cv::Scalar(127.5,127.5,127.5),true);
    // Forward
    std::vector<Mat> output_blobs;
    net.setInput(input_blob);
    net.forward(output_blobs, output_names);
    // Post process
    Mat faces = postProcess(output_blobs); // write implementation
    std::vector<Rect> boxes;
    boxes.reserve(32);
    for (int i = 0; i < faces.rows; i++) {
        float left   = (faces.at<float>(i, 0) - _oshift.x) / _sX;
        float top    = (faces.at<float>(i, 1) - _oshift.y) / _sY;
        float width  = (faces.at<float>(i, 2) - _oshift.x) / _sX - left;
        float height = (faces.at<float>(i, 3) - _oshift.y) / _sY - top;
        boxes.push_back( Rect(static_cast<int>(left + width*getXShift()  - width*(getXPortion()  - 1.0f)/2.0f),
                             static_cast<int>(top  + height*getYShift() - height*(getYPortion() - 1.0f)/2.0f),
                             static_cast<int>(width*getXPortion()),
                             static_cast<int>(height*getYPortion())) );
    }
    list = std::vector<std::vector<Point2f>>();
    list.reserve(boxes.size());
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
    return boxes;
}

std::vector<std::vector<cv::Point2f> > RetinaFaceDetector::getLandmarks() const
{
    return list;
}

cv::Mat create_anchor_centers(int height, int width, float stride, int num_anchors) {
    // Create a grid of points
    cv::Mat x_grid, y_grid;
    x_grid.create(height, width, CV_32F);
    y_grid.create(height, width, CV_32F);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            x_grid.at<float>(i, j) = j * stride;  // X coordinate (width dimension)
            y_grid.at<float>(i, j) = i * stride;  // Y coordinate (height dimension)
        }
    }

    // Combine into a 2-channel matrix and reshape to (height*width, 2)
    std::vector<cv::Mat> channels{ x_grid, y_grid };
    cv::Mat anchor_centers;
    cv::merge(channels, anchor_centers);
    anchor_centers = anchor_centers.reshape(1, height * width);

    // Replicate for multiple anchors if needed
    if (num_anchors > 1) {
        cv::Mat expanded = cv::Mat::zeros(anchor_centers.rows, num_anchors * 2, CV_32F);
        for (int i = 0; i < anchor_centers.rows; ++i) {
            for (int a = 0; a < num_anchors; ++a) {
                anchor_centers.row(i).copyTo(
                    expanded.row(i).colRange(a * 2, (a + 1) * 2)
                    );
            }
        }
        anchor_centers = expanded.reshape(1, expanded.rows * num_anchors);
    }

    return anchor_centers;
}

std::vector<Mat> RetinaFaceDetector::precomputeAnchorCenters() const {
    std::vector<Mat> _anchorCenters;
    for (int stride : feat_stride_fpn)
        _anchorCenters.push_back(create_anchor_centers(inputH() / stride, inputW() / stride, stride, num_anchors));
    return _anchorCenters;
}

// Decode bounding boxes from distances
cv::Mat RetinaFaceDetector::distance2bbox(const Mat& points, const Mat& distance) const {
    Mat x1 = points.col(0) - distance.col(0);
    Mat y1 = points.col(1) - distance.col(1);
    Mat x2 = points.col(0) + distance.col(2);
    Mat y2 = points.col(1) + distance.col(3);
    // Clip to image boundaries
    x1 = max(0, min(x1, inputW()));
    y1 = max(0, min(y1, inputH()));
    x2 = max(0, min(x2, inputW()));
    y2 = max(0, min(y2, inputH()));
    Mat bboxes;
    hconcat(std::vector<Mat>{x1, y1, x2, y2}, bboxes);
    return bboxes;
}

// Decode keypoints from distances
Mat RetinaFaceDetector::distance2kps(const Mat& points, const Mat& distance) const {
    std::vector<Mat> preds;
    for (int i = 0; i < distance.cols; i += 2) {
        Mat px = points.col(i % 2) + distance.col(i);
        Mat py = points.col(i % 2 + 1) + distance.col(i + 1);
        // Clip to image boundaries
        px = max(0, min(px, inputW()));
        py = max(0, min(py, inputH()));
        preds.push_back(px);
        preds.push_back(py);
    }
    Mat kps;
    hconcat(preds, kps);
    return kps;
}


cv::Mat gather(const cv::Mat &src, const cv::Mat &indx) {
    cv::Mat out = cv::Mat(indx.rows,src.cols,src.type());
    for(int i = 0; i < indx.rows; ++i) {
        const int *row = indx.ptr<int>(i);
        const float *source = src.ptr<float>(row[1]);
        float *target = out.ptr<float>(i);
        for(int j = 0; j < out.cols; ++j)
            target[j] = source[j];
    }
    return out;
}


cv::Mat RetinaFaceDetector::postProcess(std::vector<Mat> &outputBlobs) const
{
    std::vector<Mat> scoresList, bboxesList, kpssList;

    for (size_t idx = 0; idx < feat_stride_fpn.size(); idx++) {
        int stride = feat_stride_fpn[idx];
        Mat scores = outputBlobs[idx];
        Mat bboxPreds = outputBlobs[idx + fmc];
        Mat kpsPreds = outputBlobs[idx + fmc * 2];
        // Scale predictions
        bboxPreds *= stride;
        kpsPreds *= stride;

        Mat mask = scores >= scoreThreshold;
        cv::Mat posInds;
        findNonZero(mask, posInds); // returns [(x,y), (x,y), ...], so for 1 dimensional mask all x will be 0
        if (posInds.empty()) continue;

        // Decode boxes and keypoints
        Mat bboxes = distance2bbox(anchorCenters[idx], bboxPreds);
        Mat kpss = distance2kps(anchorCenters[idx], kpsPreds);

        // Gather highest scoored predictions
        scoresList.push_back(gather(scores,posInds));
        bboxesList.push_back(gather(bboxes,posInds));
        kpssList.push_back(gather(kpss,posInds));
    }

    // Combine all detections
    Mat allScores, allBboxes, allKpss;
    vconcat(scoresList, allScores);
    vconcat(bboxesList, allBboxes);
    vconcat(kpssList, allKpss);

    if (allBboxes.empty()) {
        return Mat(); // No detections
    }

    // Convert to Rect format for NMSBoxes
    std::vector<Rect> bboxRects;
    std::vector<float> scoresVec;
    for (int i = 0; i < allBboxes.rows; i++) {
        float x1 = allBboxes.at<float>(i, 0);
        float y1 = allBboxes.at<float>(i, 1);
        float x2 = allBboxes.at<float>(i, 2);
        float y2 = allBboxes.at<float>(i, 3);
        bboxRects.emplace_back(Point2f(x1, y1), Point2f(x2, y2));
        scoresVec.push_back(allScores.at<float>(i));
    }

    // Apply NMS using OpenCV's optimized function
    std::vector<int> keep;
    cv::dnn::NMSBoxes(bboxRects, scoresVec, scoreThreshold, nmsThreshold, keep, 1.f, topK);

    // Create output matrix with kept detections
    Mat detections(keep.size(), 4 + 10 + 1, CV_32F); // x1, y1, x2, y2, score, left_eye(x,y), right_eye(x,y), ...
    for (size_t i = 0; i < keep.size(); i++) {
        int idx = keep[i];
        detections.at<float>(i, 0) = allBboxes.at<float>(idx, 0);
        detections.at<float>(i, 1) = allBboxes.at<float>(idx, 1);
        detections.at<float>(i, 2) = allBboxes.at<float>(idx, 2);
        detections.at<float>(i, 3) = allBboxes.at<float>(idx, 3);
        for (int j = 0; j < 5; j++) {
            detections.at<float>(i, 4 + j*2) = allKpss.at<float>(idx, j * 2);
            detections.at<float>(i, 4 + j*2 + 1) = allKpss.at<float>(idx, j * 2 + 1);
        }
        detections.at<float>(i, 14) = allScores.at<float>(idx);
    }
    return detections;
}

Mat RetinaFaceDetector::resizeAndPasteInCenterOfCanvas(const Mat &_img, const Size &_canvassize, cv::Point2f &_originshift, float &_scaleX, float &_scaleY) const
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

Ptr<FaceDetector> RetinaFaceDetector::create(const std::string &_modelfilename, int _inputW, int _inputH, float _confidenceThreshold)
{
    return makePtr<RetinaFaceDetector>(_inputW,_inputH,_modelfilename,_confidenceThreshold);
}


}}

