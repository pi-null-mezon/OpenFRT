#include "facextractionutils.h"

#include <opencv2/imgproc.hpp>

std::vector<std::vector<cv::Point2f>> detectFacesLandmarks(const cv::Mat &_rgbmat, cv::Ptr<cv::ofrt::FaceDetector> &facedetector, cv::Ptr<cv::ofrt::Facemark> &facelandmarker)
{
    std::vector<std::vector<cv::Point2f>> _vlandmarks;
    const std::vector<cv::Rect> _facesboxes = facedetector->detectFaces(_rgbmat);

    std::vector<cv::Rect> _vfacesrects;
    _vfacesrects.reserve(_facesboxes.size());

    const cv::Rect framerect(0,0,_rgbmat.cols,_rgbmat.rows);
    for(const cv::Rect &_facebox: _facesboxes)
        _vfacesrects.push_back(cv::ofrt::Facemark::prepareRect(_facebox,framerect,1.7f));

    facelandmarker->fit(_rgbmat,_vfacesrects,_vlandmarks);
    return _vlandmarks;
}
