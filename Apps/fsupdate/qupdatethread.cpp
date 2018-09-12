#include "qupdatethread.h"

#include <QFile>

#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QHttpMultiPart>

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
    qInfo("labelinfo: %s",labelinfo.toUtf8().constData());
    QHttpMultiPart *multiPart = new QHttpMultiPart(QHttpMultiPart::FormDataType);

    QHttpPart labelinfoPart;
    labelinfoPart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; name=\"labelinfo\""));
    labelinfoPart.setBody(labelinfo.toLocal8Bit());
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
    qInfo("POST %s",apiurl.toUtf8().constData());
    QNetworkRequest request(QUrl::fromUserInput(apiurl));
    QNetworkAccessManager manager;
    QNetworkReply *reply = manager.post(request, multiPart);
    connect(reply,SIGNAL(finished()),this,SLOT(quit()));
    exec();

    qInfo("%s", reply->readAll().constData());

    multiPart->setParent(reply); // delete the multiPart with the reply
    reply->deleteLater();
}
