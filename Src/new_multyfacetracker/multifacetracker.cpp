#include "multifacetracker.h"

namespace cv { namespace ofrt {

MultiFaceTracker::MultiFaceTracker(const cv::Ptr<FaceDetector> &_ptr, uint _historyframes) :
    dPtr(_ptr),
    historyframes(_historyframes)
{

}

TrackedFace::TrackedFace(size_t _historylength)
{
    vhistoryrects.resize(_historylength);
    resetHistory();
    clearMetadata();
}

void TrackedFace::resetHistory()
{
    pos = 0;
    for(size_t i = 0; i < vhistoryrects.size(); ++i) {
        vhistoryrects[i] = cv::Rect(0,0,0,0);
    }
}

void TrackedFace::clearMetadata()
{
    metaId = -1;
    posted2Srv = false;
    uuid = 0;
}

}}
