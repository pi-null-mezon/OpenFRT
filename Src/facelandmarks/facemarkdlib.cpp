#include "facemarkdlib.h"

#include <dlib/opencv.h>

namespace cv { namespace face {

FacemarkDlib::FacemarkDlib() : Facemark()
{

}

void FacemarkDlib::loadModel(String model)
{
    try {
        dlib::deserialize(model.c_str()) >> shapepredictor;
    } catch(const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
}

bool FacemarkDlib::fit(InputArray image, InputArray faces, OutputArrayOfArrays landmarks)
{
    if(image.empty() || (faces.total() < 1))
        return false;

    const std::vector<cv::Rect> &_faces = *reinterpret_cast<const std::vector<cv::Rect> *>(faces.getObj());
    std::vector<std::vector<cv::Point2f>> _landmarks;
    _landmarks.reserve(_faces.size());

    dlib::cv_image<dlib::rgb_pixel> _image(image.getMat());

    for(const auto &_rect : _faces) {
        dlib::rectangle _dlibrect(static_cast<long>(_rect.x),
                                  static_cast<long>(_rect.y),
                                  static_cast<long>(_rect.x + _rect.width),
                                  static_cast<long>(_rect.y + _rect.height));
        auto _dlibshape = shapepredictor(_image, _dlibrect);
        std::vector<cv::Point2f> _points;
        _points.reserve(_dlibshape.num_parts());
        for(unsigned long i = 0; i < _dlibshape.num_parts(); ++i)
            _points.push_back(cv::Point2f(_dlibshape.part(i).x(),_dlibshape.part(i).y()));
        _landmarks.push_back(std::move(_points));
    }

    // Let's assign results to output array
    landmarks.create(static_cast<int>(_landmarks.size()), 1, CV_32FC2);
    for(size_t i = 0; i < _landmarks.size(); ++i) {
        landmarks.create(static_cast<int>(_landmarks[0].size()), 1, CV_32FC2, static_cast<int>(i));
        Mat m = landmarks.getMat(static_cast<int>(i));
        Mat(Mat(_landmarks[i]).t()).copyTo(m);
    }
    return true;
}

Ptr<Facemark> createFacemarkDlib()
{
    return makePtr<FacemarkDlib>();
}

}}
