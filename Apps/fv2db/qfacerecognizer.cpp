#include "qfacerecognizer.h"

#include "qidentificationtaskposter.h"

QFaceRecognizer::QFaceRecognizer(const QString &_oirtwebsrvurl, QObject *parent) : QObject(parent),
    oirtwebsrvurl(_oirtwebsrvurl)
{
}

void QFaceRecognizer::predict(const cv::Mat &_facemat, unsigned long _uuid) const
{
    QIdentificationTaskPoster *_thread = new QIdentificationTaskPoster(oirtwebsrvurl,_facemat,_uuid);
    connect(_thread,SIGNAL(labelPredicted(int,double,cv::String,unsigned long)),this,SIGNAL(labelPredicted(int,double,cv::String,unsigned long)));
    connect(_thread,SIGNAL(labelPredicted(int,double,cv::String,cv::Mat)),this,SIGNAL(labelPredicted(int,double,cv::String,cv::Mat)));
    connect(_thread,SIGNAL(finished()),_thread,SLOT(deleteLater()));
    _thread->start();
}
