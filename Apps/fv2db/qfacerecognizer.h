#ifndef QFACERECOGNIZER_H
#define QFACERECOGNIZER_H

#include <QObject>

#include <opencv2/core.hpp>

class QFaceRecognizer : public QObject
{
    Q_OBJECT
public:
    explicit QFaceRecognizer(const QString &_oirtwebsrvurl, QObject *parent = nullptr);

signals:
    void labelPredicted(int _label, double _distance, const cv::String &_labelInfo, unsigned long _uuid);
    void labelPredicted(int _label, double _distance, const cv::String &_labelInfo, const cv::Mat &_faceimg);

public slots:
    void predict(const cv::Mat &_facemat, unsigned long _uuid) const;

private:
    QString oirtwebsrvurl;
};

#endif // QFACERECOGNIZER_H
