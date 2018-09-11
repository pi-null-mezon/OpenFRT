#ifndef QUPDATETHREAD_H
#define QUPDATETHREAD_H

#include <QThread>

class QUpdateThread : public QThread
{
    Q_OBJECT
public:
    QUpdateThread(QObject *_parent=nullptr);
    ~QUpdateThread();

protected:
    void run();
};

#endif // QUPDATETHREAD_H
