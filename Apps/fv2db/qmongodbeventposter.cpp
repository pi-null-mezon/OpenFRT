#include "qmongodbeventposter.h"

#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>


QMongoDBEventPoster::QMongoDBEventPoster(const QString &_url, const QString &_token, const QByteArray &_data, QObject *_parent) : QThread(_parent),
    url(_url),
    token(_token),
    data(_data)
{
}

void QMongoDBEventPoster::run()
{
    QNetworkRequest _request(QUrl::fromUserInput(url));
    _request.setHeader(QNetworkRequest::ContentTypeHeader,"application/json");
    _request.setRawHeader("Authorization", QString("Bearer %1").arg(token).toUtf8());
    QNetworkAccessManager _nwmanager;
    QNetworkReply *_reply = _nwmanager.post(_request,data);
    connect(_reply,SIGNAL(finished()),this,SLOT(quit()));
    exec();
    _reply->deleteLater();
}

void postEventToMonngoDB(const QString &_url, const QString &_token, const QByteArray &_data)
{
    QMongoDBEventPoster *_thread = new QMongoDBEventPoster(_url,_token,_data);
    QObject::connect(_thread,SIGNAL(finished()),_thread,SLOT(deleteLater()));
    _thread->start();
}
