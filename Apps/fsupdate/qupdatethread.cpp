#include "qupdatethread.h"

#include <QFile>
#include <QEventLoop>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QHttpMultiPart>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>

QUpdateThread::QUpdateThread(unsigned int *_threadcounter, const QString &_apiurl, TaskType _task, const QString &_labelinfo, const QString &_filename, QObject *_parent): QThread(_parent),
    threadcounter(_threadcounter),
    apiurl(_apiurl),
    task(_task),
    labelinfo(_labelinfo),
    filename(_filename)
{
    (*threadcounter)++;
}

QUpdateThread::~QUpdateThread()
{
    (*threadcounter)--;
}

void QUpdateThread::run()
{
    QHttpMultiPart *multiPart = new QHttpMultiPart(QHttpMultiPart::FormDataType);

    QHttpPart labelinfoPart;
    labelinfoPart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; name=\"labelinfo\""));
    labelinfoPart.setBody(labelinfo.toUtf8());
    multiPart->append(labelinfoPart);

    QHttpPart imagePart;

    switch(task) {
        case Remember: {
            apiurl.append("/remember");
            imagePart.setHeader(QNetworkRequest::ContentTypeHeader, QVariant("image/jpeg"));
            imagePart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; name=\"file\"; filename=\"image.jpg\""));
            QFile *file = new QFile(filename);
            file->open(QIODevice::ReadOnly);
            imagePart.setBodyDevice(file);
            file->setParent(multiPart); // we cannot delete the file now, so delete it with the multiPart
            multiPart->append(imagePart);
        } break;

        case Delete:
            apiurl.append("/delete");
        break;

    }

    QNetworkRequest request(QUrl::fromUserInput(apiurl));
    QNetworkAccessManager manager;
    QNetworkReply *reply = manager.post(request, multiPart);
    connect(reply,SIGNAL(finished()),this,SLOT(quit()));
    exec();
    qInfo("Code: %s\n%s", reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toString().toUtf8().constData(),
          reply->readAll().constData());
    multiPart->setParent(reply); // delete the multiPart with the reply
    reply->deleteLater();
}

QStringList askLabelsInfoFrom(const QString &_apiurl)
{
    QEventLoop _el;
    QByteArray _jsondata;
    QAskLabelsThread *_thread = new QAskLabelsThread(_apiurl,&_jsondata);
    QObject::connect(_thread,SIGNAL(finished()),_thread,SLOT(deleteLater()));
    QObject::connect(_thread,SIGNAL(finished()),&_el,SLOT(quit()));
    _thread->start();
    _el.exec();

    qInfo("%s",_jsondata.constData());
    QJsonArray _jalabels = QJsonDocument::fromJson(_jsondata).object().value("labels").toArray();
    QStringList _labelsinfolist;
    _labelsinfolist.reserve(_jalabels.size());
    for(int i = 0; i < _jalabels.size(); ++i) {
        _labelsinfolist.push_back(_jalabels.at(i).toObject().value("labelinfo").toString());
    }
    return _labelsinfolist;
}

QAskLabelsThread::QAskLabelsThread(const QString &_apiurl, QByteArray *_replydata, QObject *_parent) : QThread(_parent),
    apiurl(_apiurl),
    replydata(_replydata)
{
}

void QAskLabelsThread::run()
{
    QNetworkRequest _request(QUrl::fromUserInput(apiurl.append("/labels")));
    QNetworkAccessManager _manager;
    QNetworkReply *_reply = _manager.get(_request);
    connect(_reply,SIGNAL(finished()),this,SLOT(quit()));
    exec();
    *replydata = qMove(_reply->readAll());
    _reply->deleteLater();
}
