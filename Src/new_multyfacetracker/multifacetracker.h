#ifndef MULTIFACETRACKER_H
#define MULTIFACETRACKER_H

#include "facedetector.h"

namespace cv { namespace ofrt {

class MultiFaceTracker
{
public:
    MultiFaceTracker(const cv::Ptr<cv::ofrt::FaceDetector> &_ptr, uint _historyframes=4);
    /**
        * @brief Search faces, then crop, resize and return them
        * @param Img - input image
        * @param size - output images size
        * @return vector of resized faces images
        */
    std::vector<cv::Mat> getResizedFaceImages(const cv::Mat &Img, const cv::Size size);

private:
    cv::Ptr<cv::ofrt::FaceDetector> dPtr;
    uint historyframes;

    float xPortion;
    float yPortion;
    float xShift;
    float yShift;
};

class TrackedFace
{
public:
    TrackedFace(size_t _historylength);

    void resetHistory();
    void clearMetadata();
    void updatePosition(const cv::Rect &_brect);
    void setMetaData(int _id, double _confidence, const cv::String &_info);
    cv::RotatedRect getRotatedRect() const;

private:
    // Here we will store history of face positions
    std::vector<cv::Rect> vhistoryrects;
    int                   pos;
    // Helpers
    bool        posted2Srv; // posted to identification server
    size_t      framesTracked; // how long face is tracked
    // This params will be used to store recognition result for face
    cv::String  metaInfo;
    int         metaId;
    double      metaDistance;
    // This param will be used to identify face recognition task
    size_t      uuid;
    // Face coords holder
    cv::RotatedRect rrect;
};

}}

#endif // MULTIFACETRACKER_H
