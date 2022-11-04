#include "facemarklitecnn.h"

#include <opencv2/imgproc.hpp>

static dlib::matrix<dlib::rgb_pixel> cvmat2dlibmatrix(const cv::Mat &_cvmat)
{
    cv::Mat _mat = _cvmat;
    if(_cvmat.isContinuous() == false)
        _mat = _cvmat.clone();
    unsigned char *_p = _mat.ptr<unsigned char>(0);
    dlib::matrix<dlib::rgb_pixel> _img(_mat.rows,_mat.cols);
    for(long i = 0; i < static_cast<long>(_mat.total()); ++i)
        _img(i) = dlib::rgb_pixel(_p[3*i+2],_p[3*i+1],_p[3*i]); // BGR to RGB
    return _img;
}


static cv::Mat cropInsideFromCenterAndResize(const cv::Mat &input, const cv::Size &size, cv::Rect2f &roiRect)
{
    roiRect = cv::Rect2f(0,0,0,0);
    if(static_cast<float>(input.cols)/input.rows > static_cast<float>(size.width)/size.height) {
        roiRect.height = static_cast<float>(input.rows);
        roiRect.width = input.rows * static_cast<float>(size.width)/size.height;
        roiRect.x = (input.cols - roiRect.width)/2.0f;
    } else {
        roiRect.width = static_cast<float>(input.cols);
        roiRect.height = input.cols * static_cast<float>(size.height)/size.width;
        roiRect.y = (input.rows - roiRect.height)/2.0f;
    }
    roiRect &= cv::Rect2f(0.0f, 0.0f, static_cast<float>(input.cols), static_cast<float>(input.rows));
    cv::Mat output;
    if(roiRect.area() > 0)  {
        cv::Mat croppedImg(input, roiRect);
        int interpolationMethod = cv::INTER_AREA;
        if(size.area() > roiRect.area())
            interpolationMethod = cv::INTER_CUBIC;
        cv::resize(croppedImg, output, size, 0, 0, interpolationMethod);
    }
    return output;
}


namespace cv { namespace ofrt {

FacemarkLiteCNN::FacemarkLiteCNN(const String &model) :
    Facemark(),
    isize(cv::Size(80,80))
{
    try {
        dlib::deserialize(model.c_str()) >> net;
    } catch(const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
}

bool FacemarkLiteCNN::fit(const cv::Mat &image, const std::vector<Rect> &faces, std::vector<std::vector<Point2f>> &landmarks) const
{
    if(image.empty() || (faces.size() < 1))
        return false;
    landmarks.reserve(faces.size());
    const cv::Rect frame(0,0,image.cols,image.rows);
    for(const auto &_rect : faces) {
        cv::Rect2f _roirect;
        dlib::matrix<float> prediction = net(cvmat2dlibmatrix(cropInsideFromCenterAndResize(image(_rect & frame),isize,_roirect)));
        std::vector<cv::Point2f> _points;
        _points.reserve(dlib::num_rows(prediction));
        for(long i = 0; i < dlib::num_rows(prediction)/2; ++i)
            _points.push_back(cv::Point2f((0.5f+prediction(2*i)) * _roirect.width + _rect.x + _roirect.x,
                                          (0.5f+prediction(2*i+1)) * _roirect.height + _rect.y + _roirect.y));
        landmarks.push_back(std::move(_points));
    }
    return true;
}

Ptr<Facemark> FacemarkLiteCNN::create(const String &model)
{
    return makePtr<FacemarkLiteCNN>(model);
}

}}
