#ifndef MULTIFACETRACKER_H
#define MULTIFACETRACKER_H

#include "facedetector.h"

namespace cv { namespace ofrt {

class MultiFaceTracker
{
public:
    MultiFaceTracker();

    void setFaceDetector(const cv::Ptr<cv::ofrt::FaceDetector> &_ptr);
    void enrollFrame(InputArray _image);

private:
    cv::Ptr<cv::ofrt::FaceDetector> dPtr;
};

}}

#endif // MULTIFACETRACKER_H
