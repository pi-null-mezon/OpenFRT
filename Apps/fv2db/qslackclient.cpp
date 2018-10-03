#include "qslackclient.h"

#include "qslackimageposter.h"

QSlackClient::QSlackClient(QObject *_parent) : QObject(_parent),
    unrecfacesinrow(0)
{
}

void QSlackClient::enrollRecognition(int _label, double _distance, const cv::String &_labelInfo, const cv::Mat &_img)
{
    QTime _time = QTime::currentTime();
    if(_label > -1) {
        if((prevLabel != _label) || (prevTime.secsTo(_time) > 7)) {           
            postImageIntoSlackChannel(getSlackchannelid(),getSlackbottoken(),_label,_distance,_labelInfo,_img);
        }
        unrecfacesinrow = 0;
        prevLabel = _label;
    } else {
        if(prevTime.secsTo(_time) > 11) {
            unrecfacesinrow = 1;
        } else {
            unrecfacesinrow++;
        }
    }

    prevTime  = _time;

    if(unrecfacesinrow == 15) {
        postImageIntoSlackChannel(getSlackchannelid(),getSlackbottoken(),-1,-1.0,"Незнакомец",_img);
    }
}

QString QSlackClient::getSlackchannelid() const
{
    return slackchannelid;
}

void QSlackClient::setSlackchannelid(const QString &value)
{
    slackchannelid = value;
}

QString QSlackClient::getSlackbottoken() const
{
    return slackbottoken;
}

void QSlackClient::setSlackbottoken(const QString &value)
{
    slackbottoken = value;
}
