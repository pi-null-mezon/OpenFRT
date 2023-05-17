#ifndef FACEXTRACTIONUTILS_H
#define FACEXTRACTIONUTILS_H

#include <opencv2/core.hpp>

#include "facedetector.h"
#include "facemark.h"

std::vector<std::vector<cv::Point2f>> detectFacesLandmarks(const cv::Mat &_rgbmat, cv::Ptr<cv::ofrt::FaceDetector> &facedetector, cv::Ptr<cv::ofrt::Facemark> &facelandmarker);

#endif // FACEXTRACTIONUTILS_H
