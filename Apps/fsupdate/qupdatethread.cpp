#include "qupdatethread.h"
#include <QDebug>

QUpdateThread::QUpdateThread(QObject *_parent): QThread(_parent)
{
    qDebug() << "Constructor:";
    qDebug() << currentThread();
}

QUpdateThread::~QUpdateThread()
{
    qDebug() << "Destructor:";
    qDebug() << currentThread();
}

void QUpdateThread::run()
{
    qDebug() << "run():";
    qDebug() << currentThread();
    qDebug() << thread();
}
