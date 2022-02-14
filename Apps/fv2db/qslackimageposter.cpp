#include "qslackimageposter.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QJsonDocument>
#include <QHttpPart>

QSlackImagePoster::QSlackImagePoster(const QString &_slackchannelid, const QString &_slacktoken, int _label, double _distance, const cv::String &_labelInfo, const cv::Mat &_cvmat) :
    QThread(),
    slackchannelid(_slackchannelid),
    slacktoken(_slacktoken),
    label(_label),
    distance(_distance),
    labelinfo(_labelInfo.c_str())
{
    img = _cvmat;
}

QSlackImagePoster::~QSlackImagePoster()
{
    //qInfo("~QSlackImagePoster");
}

void QSlackImagePoster::run()
{
    QHttpMultiPart *fields = new QHttpMultiPart(QHttpMultiPart::FormDataType);
    QHttpPart photo;
    photo.setHeader(QNetworkRequest::ContentTypeHeader, QVariant("image/jpeg"));
    photo.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant(QString("form-data; name=\"file\"; filename=\"%1.jpg\"").arg(labelinfo))); // filename is needed here
    if(!img.empty()) {
        cv::resize(img,img,cv::Size(78,92),0,0,cv::INTER_AREA);
        std::vector<uchar> _vbytes;
        cv::imencode(".jpg",img,_vbytes);
        photo.setBody(QByteArray((const char*)_vbytes.data(), (int)_vbytes.size()));
    }
    fields->append(photo);

    QString _comment = QString("Distance: %1").arg(QString::number(distance,'f',3));
    QString _requeststring = QString("https://slack.com/api/files.upload?token=%1&channels=%2&file&initial_comment=%3").arg(slacktoken,slackchannelid,_comment);

    QNetworkAccessManager netmgr;
    QNetworkRequest req(_requeststring);

    QNetworkReply *reply = netmgr.post(req, fields);
    QObject::connect(reply, SIGNAL(finished()), this, SLOT(quit()));
    exec();

    if(reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt() == 200) {
        //qInfo("[QSlackImagePoster] Reply: %s", reply->readAll().constData());
    } else {
        //qInfo("[QSlackImagePoster] Error: %s", reply->errorString().toUtf8().constData());
    }

    fields->setParent(reply);
    reply->deleteLater();
}

void postImageIntoSlackChannel(const QString &_slackchannelid, const QString &_slacktoken, int _label, double _distance, const cv::String &_labelInfo, const cv::Mat &_cvmat)
{
    QSlackImagePoster *_workerthread = new QSlackImagePoster(_slackchannelid,_slacktoken,_label,_distance,_labelInfo,_cvmat);
    QObject::connect(_workerthread,SIGNAL(finished()),_workerthread,SLOT(deleteLater()));
    _workerthread->start();
}
