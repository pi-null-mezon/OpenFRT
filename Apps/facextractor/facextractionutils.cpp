#include "facextractionutils.h"

#include <opencv2/imgproc.hpp>

cv::Rect squareRectFromCenter(const cv::Rect &_source)
{
    if(_source.width == _source.height)
        return _source;
    else if(_source.width > _source.height)
        return cv::Rect(_source.x + (_source.width - _source.height) / 2, _source.y,_source.height, _source.height);
    return cv::Rect(_source.x, _source.y + (_source.height - _source.width) / 2, _source.width, _source.width);
}

std::vector<std::vector<cv::Point2f>> detectFacesLandmarks(const cv::Mat &_rgbmat, cv::Ptr<cv::ofrt::FaceDetector> &facedetector, cv::Ptr<cv::ofrt::Facemark> &facelandmarker)
{
    std::vector<std::vector<cv::Point2f>> _vlandmarks;
    const std::vector<cv::Rect> _facesboxes = facedetector->detectFaces(_rgbmat);
    const float _upscalefactor = 1.4f;
    const cv::Rect framerect(0,0,_rgbmat.cols,_rgbmat.rows);
    std::vector<cv::Rect> _vfacesrects;
    _vfacesrects.reserve(_facesboxes.size());
    for(const cv::Rect &_facebox: _facesboxes) {
        const cv::Rect _facerect = squareRectFromCenter(_facebox);
        cv::Rect _facerectforlandmarks = cv::Rect(_facerect.x - _facerect.width * (_upscalefactor - 1.0f) / 2.0f,
                                                  _facerect.y - _facerect.height * (_upscalefactor - 1.0f) / 2.0f,
                                                  _facerect.width*_upscalefactor,
                                                  _facerect.height*_upscalefactor) & framerect;
        _vfacesrects.push_back(std::move(_facerectforlandmarks));
    }
    facelandmarker->fit(_rgbmat,_vfacesrects,_vlandmarks);
    return _vlandmarks;
}
