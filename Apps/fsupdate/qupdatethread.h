#ifndef QUPDATETHREAD_H
#define QUPDATETHREAD_H

#include <QThread>

class QUpdateThread : public QThread
{
    Q_OBJECT
public:
    enum TaskType {Unknown, Remember, Delete};

    QUpdateThread(unsigned int *_threadcounter, const QString &_apiurl, TaskType _task, const QString &_labelinfo, const QString &_filename=QString(), QObject *_parent=nullptr);
    ~QUpdateThread();

protected:
    void run();

private:
    QString  apiurl;
    QString  filename;
    QString  labelinfo;
    TaskType task;
    unsigned int *threadcounter;
};

#endif // QUPDATETHREAD_H
