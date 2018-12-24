#ifndef MULTIFACETRACKER_H
#define MULTIFACETRACKER_H

#include "facedetector.h"

namespace cv { namespace ofrt {

/**
 * @brief The TrackedFace class
 */
class TrackedFace
{
public:
    TrackedFace(int _historylength=8); // controls how long face should not be dropped

    void clearMetadata();
    void updatePosition(const cv::Rect &_brect);
    void decreaseFramesTracked();
    void setMetaData(int _id, double _distance, const cv::String &_info);
    void setPosted2Srv(bool _value);
    cv::Rect getRect(int _averagelast) const;


    int getFramesTracked() const;

    unsigned long getUuid() const;
    void setUuid(unsigned long value);

private:

    std::vector<cv::Rect>   vhistoryrects;  // history of face positions
    int                     historylength;  // length of history
    int                     pos;            // current position in vhistoryrects
    bool                    posted2Srv;     // posted to identification server
    int                     framesTracked;  // how long face is tracked
    cv::String              metaInfo;
    int                     metaId;
    double                  metaDistance;
    unsigned long           uuid;           // unique identifier of the tracked face
};

/**
 * @brief The MultiFaceTracker class
 */
class MultiFaceTracker
{
public:
    MultiFaceTracker(const cv::Ptr<cv::ofrt::FaceDetector> &_ptr, size_t maxfaces=4);
    /**
     * @brief Search faces, then crop, resize and return them
     * @param Img - input image
     * @param size - output images size
     * @return vector of resized faces images
     */
    std::vector<cv::Mat> getResizedFaceImages(const cv::Mat &_img, const Size &_size, int _averagelast=4);
    /**
     * @brief getTrackedFaces get information about tracked faces
     * @return vector of tracked faces
     */
    std::vector<TrackedFace> getTrackedFaces() const;



private:
    cv::Mat         __cropInsideFromCenterAndResize(const cv::Mat &input, const cv::Size &size);
    void            __enrollImage(const cv::Mat &_img);
    unsigned long   __nextUUID();

    cv::Ptr<cv::ofrt::FaceDetector> dPtr;    
    std::vector<TrackedFace> vtrackedfaces;

    unsigned long uuid;
};

}}

#endif // MULTIFACETRACKER_H
