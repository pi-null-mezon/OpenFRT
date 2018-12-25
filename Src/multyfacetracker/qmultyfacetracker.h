#ifndef QMULTYFACETRACKER_H
#define QMULTYFACETRACKER_H

#include <QObject>

#include "multyfacetracker.h"

class QMultyFaceTracker : public QObject
{
    Q_OBJECT
public:
    explicit QMultyFaceTracker(const cv::Ptr<cv::ofrt::FaceDetector> &_cvptrfacedet, uint _maxfaces, QObject *parent=nullptr);
    void setTargetFaceSize(const cv::Size &size);
    void enableVisualization(bool _value);

signals:
    void faceWithoutLabelFound(const cv::Mat &_facemat, unsigned long _uuid);
    void frameProcessed();

public slots:
    void enrollImage(const cv::Mat &inputImg);
    void setLabelForTheFace(int _id, double _distance, const cv::String &_info, unsigned long _uuid);

private:
    cv::ofrt::MultyFaceTracker multyfacetracker;
    cv::Size targetSize;
    bool visualization;
};

/**
 * @brief As cv::putText(...) can not draw cyrillic gliphs we should convert them into latin, this function performs such transformation
 * @param _cvcyrstr - input string, it can contain latin or cyrrilic symbols in UTF8 code
 * @return string with latin representation of the input string
 */
cv::String utf8cyr2utf8latin(const cv::String &_cvcyrstr);

#endif // QMULTYFACETRACKER_H
