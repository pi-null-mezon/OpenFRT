#include "qmongodbclient.h"

#include <QDateTime>

#include "qmongodbeventposter.h"

QMongoDBClient::QMongoDBClient(QObject *parent) : QObject(parent)
{
}

QString QMongoDBClient::getUrl() const
{
    return url;
}

void QMongoDBClient::setUrl(const QString &value)
{
    url = value;
}

QString QMongoDBClient::getToken() const
{
    return token;
}

void QMongoDBClient::setToken(const QString &value)
{
    token = value;
}

void QMongoDBClient::enrollRecognition(int _label, double _distance, const cv::String &_labelInfo, const cv::Mat &_img)
{
    QTime _time = QTime::currentTime();
    if(_label > -1) {
        if((prevLabel != _label) || (prevTime.secsTo(_time) > 7)) {
            // TO DO - make JSON for event
            //postEventToMonngoDB(getUrl(),getToken());


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
    if(unrecfacesinrow == 15) {
        // TO DO - make JSON for event
       // postEventToMonngoDB(getUrl(),getToken());
    }
    prevTime  = _time;
}
