#ifndef QMONGODBCLIENT_H
#define QMONGODBCLIENT_H

#include <QObject>
#include <QDateTime>

#include <opencv2/core.hpp>

class QMongoDBClient : public QObject
{
    Q_OBJECT
public:
    explicit QMongoDBClient(QObject *parent = nullptr);

    QString getUrl() const;
    void setUrl(const QString &value);

    QString getToken() const;
    void setToken(const QString &value);

    QString getSpotid() const;
    void setSpotid(const QString &value);

public slots:
    void enrollRecognition(int _label, double _distance, const cv::String &_labelInfo, const cv::RotatedRect &_rrect);

private:
    QTime prevTime;
    int   prevLabel;
    unsigned int unrecfacesinrow;

    QString url;
    QString token;
    QString spotid;
};

#endif // QMONGODBCLIENT_H
