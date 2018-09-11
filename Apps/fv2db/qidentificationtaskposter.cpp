#include "qidentificationtaskposter.h"

#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QHttpMultiPart>

#include <QJsonDocument>
#include <QJsonObject>

#include <opencv2/imgcodecs.hpp>

QIdentificationTaskPoster::QIdentificationTaskPoster(const QString &_urlstr, const cv::Mat &_facemat, const cv::RotatedRect &_facerr, QObject *_parent) : QThread(_parent),
    urlstr(_urlstr),
    facemat(_facemat),
    facerr(_facerr)
{
}

void QIdentificationTaskPoster::run()
{
    QHttpMultiPart *_fields = new QHttpMultiPart(QHttpMultiPart::FormDataType);

    QHttpPart _photo;
    _photo.setHeader(QNetworkRequest::ContentTypeHeader, "image/jpeg");
    _photo.setHeader(QNetworkRequest::ContentDispositionHeader, "form-data; name=\"file\"; filename=\"face.jpg\"");
    std::vector<unsigned char> _vjpegdata;    
    cv::imencode(".jpg",facemat,_vjpegdata);
    _photo.setBody(QByteArray::fromRawData((const char*)_vjpegdata.data(), static_cast<int>(_vjpegdata.size())));

    _fields->append(_photo);

    QNetworkRequest _request(QUrl::fromUserInput(urlstr));
    QNetworkAccessManager _netmgr;
    QNetworkReply *_reply = _netmgr.post(_request, _fields);   
    QObject::connect(_reply, SIGNAL(finished()), this, SLOT(quit()));
    exec();

    if(_reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt() == 200) {
        QJsonParseError _jperror;
        QByteArray _replydata = _reply->readAll();
        QJsonObject _json = QJsonDocument::fromJson(_replydata,&_jperror).object();
        if(_jperror.error == QJsonParseError::NoError) {
            if(QString::compare(_json.value("status").toString(),"Success",Qt::CaseInsensitive) == 0) { // equal strings
                qInfo("%s", _replydata.constData());
                double _distance = _json.value("distance").toDouble();
                if(_distance < _json.value("distancethresh").toDouble()) {
                    emit labelPredicted(_json.value("label").toInt(),
                                        _distance,
                                        _json.value("labelinfo").toString().toUtf8().constData(),
                                        facerr);
                } else {
                    emit labelPredicted(-1,DBL_MAX,"",facerr);
                }
            } else {
                qWarning("%s", _replydata.constData());
                emit labelPredicted(-1,DBL_MAX,"",facerr);
            }
        } else {
            qWarning("JSON parser error - %s", _jperror.errorString().toUtf8().constData());
        }
    }

    _fields->setParent(_reply);
    _reply->deleteLater();
}
