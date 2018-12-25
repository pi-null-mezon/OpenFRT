#include "qvideolocker.h"

QVideoLocker::QVideoLocker(QObject *parent) : QObject(parent),
  locked(false)
{    
}

void QVideoLocker::updateFrame(const cv::Mat &_framemat, unsigned long _uuid)
{
    if(locked == false) {
        __lock();
        emit frameUpdated(_framemat, _uuid);
    }
}

void QVideoLocker::unlock()
{
    __unlock();
}

void QVideoLocker::__lock()
{
    if(!locked) {
        locked = true;
    }
}

void QVideoLocker::__unlock()
{
    if(locked) {
        locked = false;
    }
}
