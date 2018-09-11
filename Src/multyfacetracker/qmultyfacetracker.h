#ifndef QMULTYFACETRACKER_H
#define QMULTYFACETRACKER_H

#include <QObject>
#include "multyfacetracker.h"

class QMultyFaceTracker : public QObject
{
    Q_OBJECT
public:
    explicit QMultyFaceTracker(uint _maxfaces = 7, QObject *parent=nullptr);

    int getRecognizerid() const;
    void setRecognizerid(int value);

signals:
    void faceFound(const cv::Mat &faceImg, const cv::RotatedRect &rRect);
    void faceWithoutLabelFound(const cv::Mat &_facemat, const cv::RotatedRect &_rrect);

public slots:
    void enrollImage(const cv::Mat &inputImg);
    bool setEyeClassifier(cv::CascadeClassifier *pointer);
    bool setFaceClassifier(cv::CascadeClassifier *pointer);
    void setDlibFaceShapePredictor(dlib::shape_predictor *pointer);
    void setTargetFaceSize(const cv::Size &size);
    void setMaxFaces(uint value);
    void setFaceRectPortions(float _xP, float _yP);
    void setFaceRectShifts(float _xShift, float _yShift);
    void setFaceAlignmentMethod(FaceTracker::AlignMethod _method);
    /**
     * @brief setLabelForTheFace - works same as named, particular face tracker is selected by the nearest rectangle evaluation
     * @param _id - class identifier
     * @param _confidence - self eplained (Euclidean distance in the feature space to the nearest class)
     * @param _info - string of the information about class
     * @param _rrect - target rectangle to search appropriate face tracker
     */
    void setLabelForTheFace(int _id, double _confidence, const cv::String &_info, const cv::RotatedRect &_rrect);
    size_t getMaxFaces() const;
    void setVisualization(bool _value);

    void setVerbose(bool _enable);

private:    
    MultyFaceTracker m_tracker;
    cv::Size m_targetSize;    
    bool f_visualization = true;
    bool f_verbose = false;
    /**
     * @brief m_recognizerid - stores facerecognizer identifier (should be used as the camera id to log events on the particular viewpoint)
     */
    int recognizerid = -1;
};

/**
 * @brief As cv::putText(...) can not draw cyrillic gliphs we should convert them into latin, this function performs such transformation
 * @param _cvcyrstr - input string, it can contain latin or cyrrilic symbols in UTF8 code
 * @return string with latin representation of the input string
 */
cv::String utf8cyr2utf8latin(const cv::String &_cvcyrstr);


#endif // QMULTYFACETRACKER_H
