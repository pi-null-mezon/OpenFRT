#include "facemarkcnn.h"

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


cv::Mat cropInsideFromCenterAndResize(const cv::Mat &input, const cv::Size &size, cv::Rect &roiRect)
{
    roiRect = cv::Rect(0,0,0,0);
    if(static_cast<float>(input.cols)/input.rows > static_cast<float>(size.width)/size.height) {
        roiRect.height = static_cast<float>(input.rows);
        roiRect.width = input.rows * static_cast<float>(size.width)/size.height;
        roiRect.x = (input.cols - roiRect.width)/2.0f;
    } else {
        roiRect.width = static_cast<float>(input.cols);
        roiRect.height = input.cols * static_cast<float>(size.height)/size.width;
        roiRect.y = (input.rows - roiRect.height)/2.0f;
    }
    roiRect &= cv::Rect(0, 0, input.cols, input.rows);
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


namespace cv { namespace face {

FacemarkCNN::FacemarkCNN() : Facemark(), isize(cv::Size(100,100))
{

}

void FacemarkCNN::loadModel(String model)
{
    try {
        dlib::deserialize(model.c_str()) >> net;
    } catch(const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
}

bool FacemarkCNN::fit(InputArray image, InputArray faces, OutputArrayOfArrays landmarks)
{
    if(image.empty() || (faces.total() < 1))
        return false;

    const std::vector<cv::Rect> &_faces = *reinterpret_cast<const std::vector<cv::Rect> *>(faces.getObj());
    std::vector<std::vector<cv::Point2f>> _landmarks;
    _landmarks.reserve(_faces.size());

    cv::Mat mat = image.getMat();
    for(const auto &_rect : _faces) {
        cv::Rect _roirect;
        dlib::matrix<float> prediction = net(cvmat2dlibmatrix(cropInsideFromCenterAndResize(mat(_rect & cv::Rect(0,0,mat.cols,mat.rows)),isize,_roirect)));

        std::vector<cv::Point2f> _points;
        _points.reserve(dlib::num_rows(prediction));
        for(long i = 0; i < dlib::num_rows(prediction)/2; ++i)
            _points.push_back(cv::Point2f((0.5f+prediction(2*i)) * _roirect.width + _rect.x + _roirect.x,
                                          (0.5f+prediction(2*i+1)) * _roirect.height + _rect.y + _roirect.y));
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

Ptr<Facemark> createFacemarkCNN()
{
    return makePtr<FacemarkCNN>();
}

}}
