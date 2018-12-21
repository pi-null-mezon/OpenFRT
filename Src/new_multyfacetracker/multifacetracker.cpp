#include "multifacetracker.h"

namespace cv { namespace ofrt {

MultiFaceTracker::MultiFaceTracker()
{

}

void MultiFaceTracker::setFaceDetector(const cv::Ptr<cv::ofrt::FaceDetector> &_ptr)
{
    dPtr = _ptr;
}

}}
