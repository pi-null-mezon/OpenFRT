#include "qvideolocker.h"

QVideoLocker::QVideoLocker(QObject *parent) : QObject(parent)
{    
}

void QVideoLocker::updateFrame(const cv::Mat &_framemat, const QUuid &_quuid)
{
    if(m_locked == false) {
        __lock();
        emit frameUpdated(_framemat, _quuid);
    }
}

void QVideoLocker::unlock()
{
    __unlock();
}

void QVideoLocker::__lock()
{
    if(!m_locked) {
        m_locked = true;
    }
}

void QVideoLocker::__unlock()
{
    if(m_locked) {
        m_locked = false;
    }
}
