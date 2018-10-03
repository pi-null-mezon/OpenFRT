#ifndef QMONGODBEVENTPOSTER_H
#define QMONGODBEVENTPOSTER_H

#include <QThread>

void postEventToMonngoDB(const QString &_url, const QString &_token, const QByteArray &_data);

class QMongoDBEventPoster : public QThread
{
    Q_OBJECT
public:
    QMongoDBEventPoster(const QString &_url, const QString &_token, const QByteArray &_data, QObject *_parent=nullptr);

protected:
    void run() override;

private:
    QString url;
    QString token;
    QByteArray data;
};

#endif // QMONGODBEVENTPOSTER_H
