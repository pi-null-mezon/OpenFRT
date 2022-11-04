#include "facemarkdlib.h"

#include <dlib/opencv.h>

namespace cv { namespace ofrt {

FacemarkDlib::FacemarkDlib(const String &model) :
    Facemark()
{
    try {
        dlib::deserialize(model.c_str()) >> shapepredictor;
    } catch(const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
}

bool FacemarkDlib::fit(const cv::Mat &image, const std::vector<Rect> &faces, std::vector<std::vector<Point2f>> &landmarks) const
{
    if(image.empty() || (faces.size() < 1))
        return false;

    landmarks.reserve(faces.size());

    dlib::cv_image<dlib::rgb_pixel> _image(image);
    for(const auto &_rect : faces) {
        dlib::rectangle _dlibrect(static_cast<long>(_rect.x),
                                  static_cast<long>(_rect.y),
                                  static_cast<long>(_rect.x + _rect.width),
                                  static_cast<long>(_rect.y + _rect.height));
        auto _dlibshape = shapepredictor(_image, _dlibrect);
        std::vector<cv::Point2f> _points;
        _points.reserve(_dlibshape.num_parts());
        for(unsigned long i = 0; i < _dlibshape.num_parts(); ++i)
            _points.push_back(cv::Point2f(_dlibshape.part(i).x(),_dlibshape.part(i).y()));
        landmarks.push_back(std::move(_points));
    }
    return true;
}

Ptr<Facemark> FacemarkDlib::create(const String &model)
{
    return makePtr<FacemarkDlib>(model);
}

}}

