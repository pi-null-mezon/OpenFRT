#include "qmongodbclient.h"

#include <QDateTime>
#include <QJsonObject>
#include <QJsonDocument>

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

void QMongoDBClient::enrollRecognition(int _label, double _distance, const cv::String &_labelInfo, const cv::RotatedRect &_rrect)
{
    Q_UNUSED(_distance);
    Q_UNUSED(_rrect);
    const QDateTime _dt = QDateTime::currentDateTime();
    if(_label > -1) {
        if((prevLabel != _label) || (prevTime.secsTo(_dt.time()) > 7)) {
            // TO DO - make JSON for event
            QJsonObject _json;
            _json["labelInfo"] = _labelInfo.c_str();
            _json["camid"]  = getSpotid();
            _json["camdt"]  = _dt.toString("hh:mm:ss dd.MM.yyyy");
            QJsonObject _eventjson;
            _eventjson["event"] = _json;
            postEventToMonngoDB(getUrl(),getToken(),QJsonDocument(_eventjson).toJson());
        }
        unrecfacesinrow = 0;
        prevLabel = _label;
    } else {
        if(prevTime.secsTo(_dt.time()) > 11) {
            unrecfacesinrow = 1;
        } else {
            unrecfacesinrow++;
        }
    }
    if(unrecfacesinrow == 15) {
        QJsonObject _json;
        _json["labelInfo"] = "Unknown";
        _json["camid"]  = getSpotid();
        _json["camdt"]  = _dt.toString("hh:mm:ss dd.MM.yyyy");
        QJsonObject _eventjson;
        _eventjson["event"] = _json;
        postEventToMonngoDB(getUrl(),getToken(),QJsonDocument(_eventjson).toJson());
    }
    prevTime  = _dt.time();
}

QString QMongoDBClient::getSpotid() const
{
    return spotid;
}

void QMongoDBClient::setSpotid(const QString &value)
{
    spotid = value;
}
