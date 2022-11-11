#include "facedetector.h"

namespace cv { namespace ofrt {

FaceDetector::FaceDetector() :
    xPortion(1),
    yPortion(1),
    xShift(0),
    yShift(0)
{
}

FaceDetector::~FaceDetector()
{
}

void FaceDetector::setPortions(float _xportion, float _yportion)
{
    xPortion = _xportion;
    yPortion = _yportion;
}

void FaceDetector::setShifts(float _xshift, float _yshift)
{
    xShift = _xshift;
    yShift = _yshift;
}

float FaceDetector::getXPortion() const
{
    return xPortion;
}

float FaceDetector::getYPortion() const
{
    return yPortion;
}

float FaceDetector::getXShift() const
{
    return xShift;
}

float FaceDetector::getYShift() const
{
    return yShift;
}

void FaceDetector::sortByArea(std::vector<Rect> &rects, bool descending)
{
    std::sort(rects.begin(),rects.end(),[descending](const Rect &left, const Rect &right) {
        if(descending)
            return left.area() > right.area();
        else
            return left.area() <= right.area();}
    );
}

}}

