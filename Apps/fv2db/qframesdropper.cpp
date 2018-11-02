#include "qframesdropper.h"

QFramesDropper::QFramesDropper(QObject *parent) : QObject(parent),
    drop(false)
{
}

void QFramesDropper::updateFrame(const cv::Mat &_frame)
{
    if(drop == false) {
        drop = true;
        emit frameUpdated(_frame);
    }
}

void QFramesDropper::passFrames()
{
    drop = false;
}
