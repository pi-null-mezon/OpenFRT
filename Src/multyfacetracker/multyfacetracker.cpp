#include "multyfacetracker.h"

#include <opencv2/imgproc.hpp>

namespace cv { namespace ofrt {

MultyFaceTracker::MultyFaceTracker() :
    uuid(0)
{
}

MultyFaceTracker::MultyFaceTracker(const cv::Ptr<FaceDetector> &_ptr, size_t _maxfaces) :
    dPtr(_ptr),
    uuid(0)
{
    vtrackedfaces.resize(_maxfaces);
}

std::vector<std::pair<size_t,Mat>> MultyFaceTracker::getResizedFaceImages(const Mat &_img, const Size &_size, int _averagelast)
{
    __enrollImage(_img);
    std::vector<std::pair<size_t,Mat>> _vfaces;
    _vfaces.reserve(vtrackedfaces.size());
    cv::Rect _imgboundingrect(0,0,_img.cols,_img.rows);
    for(size_t i = 0; i < vtrackedfaces.size(); ++i) {
        if(vtrackedfaces[i].getFramesTracked() > 0)
            _vfaces.push_back(std::make_pair(i,__cropInsideFromCenterAndResize(_img(vtrackedfaces[i].getRect(_averagelast) & _imgboundingrect),_size)));
    }
    return _vfaces;
}

TrackedFace *MultyFaceTracker::at(size_t i)
{
    return &vtrackedfaces[i];
}

size_t MultyFaceTracker::maxFaces() const
{
    return vtrackedfaces.size();
}

void MultyFaceTracker::setFaceDetector(const cv::Ptr<FaceDetector> &_ptr, size_t _maxfaces)
{
    dPtr = _ptr;
    vtrackedfaces.resize(_maxfaces);
}

void MultyFaceTracker::clear()
{
    for(size_t i = 0; i < vtrackedfaces.size(); ++i)
        vtrackedfaces[i] = TrackedFace();
}

Mat MultyFaceTracker::__cropInsideFromCenterAndResize(const Mat &input, const Size &size)
{
    cv::Rect2f roiRect(0,0,0,0);
    if((float)input.cols/input.rows > (float)size.width/size.height) {
        roiRect.height = (float)input.rows;
        roiRect.width = input.rows * (float)size.width/size.height;
        roiRect.x = (input.cols - roiRect.width)/2.0f;
    } else {
        roiRect.width = (float)input.cols;
        roiRect.height = input.cols * (float)size.height/size.width;
        roiRect.y = (input.rows - roiRect.height)/2.0f;
    }
    roiRect &= cv::Rect2f(0, 0, (float)input.cols, (float)input.rows);
    cv::Mat output;
    if(roiRect.area() > 0)  {
        cv::Mat croppedImg(input, roiRect);
        int interpolationMethod = CV_INTER_AREA;
        if(size.area() > roiRect.area())
            interpolationMethod = CV_INTER_CUBIC;
        cv::resize(croppedImg, output, size, 0, 0, interpolationMethod);
    }
    return output;
}

void MultyFaceTracker::__enrollImage(const Mat &_img)
{
    std::vector<cv::Rect> vrects = dPtr->detectFaces(_img);
    std::vector<bool>     alreadyused(vrects.size(),false);
    // First we need to update tracked faces
    for(size_t i = 0; i < vtrackedfaces.size(); ++i) {
        TrackedFace &_tf = vtrackedfaces[i];
        bool updated = false;
        for(size_t j = 0; j < vrects.size(); ++j) {
            if(alreadyused[j] == false) {
                if(_tf.getFramesTracked() > 0) {
                    const cv::Rect _lastrect = _tf.getRect(1);
                    if((_lastrect & vrects[j]).area() > _lastrect.area() / 2) {
                        _tf.updatePosition(vrects[j]);
                        alreadyused[j] = true;
                        updated = true;
                        break;
                    }
                }
            }
        }
        if(updated == false) {
            _tf.decreaseFramesTracked();
        }
    }
    // Second we need enroll new faces
    for(size_t i = 0; i < vrects.size(); ++i) {
        if(alreadyused[i] == false) {
            for(size_t j = 0; j < vtrackedfaces.size(); ++j) {
                TrackedFace &_tf = vtrackedfaces[j];
                if(_tf.getFramesTracked() == 0) {
                    _tf.updatePosition(vrects[i]);
                    _tf.setUuid(__nextUUID()); // generate new guid
                    break;
                }
            }
        }
    }
}

unsigned long MultyFaceTracker::__nextUUID()
{
    return ++uuid;
}


TrackedFace::TrackedFace(int _historylength) :
    pos(0),
    framesTracked(0)
{
    historylength = _historylength;
    vhistoryrects.resize(historylength,cv::Rect(0,0,0,0));
    clearMetadata();
}

void TrackedFace::clearMetadata()
{
    unknownInRow = 0;
    metaId = -1; // -1 - unknown person, -2 - error on recognition (for the instance face could not be found)
    metaInfo = "Identification...";
    metaDistance = -1;
    posted2Srv = false;
}

void TrackedFace::updatePosition(const Rect &_brect)
{
    if(framesTracked == 0) {
        pos = 0;
        for(size_t i = 0; i < vhistoryrects.size(); ++i)
            vhistoryrects[i] = _brect;
    } else {
        vhistoryrects[pos] = _brect;
    }
    framesTracked = framesTracked <= historylength ? framesTracked + 1 : historylength;
    pos = (pos + 1) % historylength;
}

void TrackedFace::decreaseFramesTracked()
{
    framesTracked = framesTracked > 0 ? framesTracked - 1 : 0;
    if(framesTracked == 0) {
        clearMetadata();
    }
}

void TrackedFace::setMetaData(int _id, double _distance, const String &_info)
{
    if(_id == -1) {
        if(metaId == -1)
            unknownInRow++;
    } else {
        unknownInRow = 0;
    }
    metaId = _id;
    metaDistance = _distance;
    metaInfo = _info;
}

Rect TrackedFace::getRect(int _averagelast) const
{
    float _x = 0.0f, _y = 0.0f, _w = 0.0f, _h = 0.0f;
    for(int i = 0; i < _averagelast; ++i) {
        const cv::Rect &_rect = vhistoryrects[ (((pos - 1 - i) % historylength) + historylength) % historylength];
        _x += _rect.x;
        _y += _rect.y;
        _w += _rect.width;
        _h += _rect.height;
    }
    _x /= _averagelast;
    _y /= _averagelast;
    _w /= _averagelast;
    _h /= _averagelast;
    return cv::Rect(_x,_y,_w,_h);
}

int TrackedFace::getFramesTracked() const
{
    return framesTracked;
}

int TrackedFace::getMetaId() const
{
    return metaId;
}

unsigned long TrackedFace::getUuid() const
{
    return uuid;
}

void TrackedFace::setUuid(unsigned long value)
{
    uuid = value;
}

bool TrackedFace::getPosted2Srv() const
{
    return posted2Srv;
}

void TrackedFace::setPosted2Srv(bool value)
{
    posted2Srv = value;
}

int TrackedFace::getUnknownInRow() const
{
    return unknownInRow;
}

double TrackedFace::getMetaDistance() const
{
    return metaDistance;
}

cv::String TrackedFace::getMetaInfo() const
{
    return metaInfo;
}

}}
