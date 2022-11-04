#ifndef FACEXTRACTIONUTILS_H
#define FACEXTRACTIONUTILS_H

#include <opencv2/core.hpp>

#include "facedetector.h"
#include "facemark.h"

std::vector<std::vector<cv::Point2f>> detectFacesLandmarks(const cv::Mat &_rgbmat, cv::Ptr<cv::ofrt::FaceDetector> &facedetector, cv::Ptr<cv::ofrt::Facemark> &facelandmarker);

cv::Mat extractFacePatch(const cv::Mat &_rgbmat, const std::vector<cv::Point2f> &_landmarks, float _targeteyesdistance, const cv::Size &_targetsize, float h2wshift=0, float v2hshift=0, bool rotate=true);

#endif // FACEXTRACTIONUTILS_H
