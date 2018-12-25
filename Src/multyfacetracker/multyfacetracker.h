#ifndef MULTYFACETRACKER_H
#define MULTYFACETRACKER_H

#include "facedetector.h"

namespace cv { namespace ofrt {

/**
 * @brief The TrackedFace class
 */
class TrackedFace
{
public:
    TrackedFace(int _historylength=5); // controls how long face tracker should not be dropped

    void clearMetadata();
    void updatePosition(const cv::Rect &_brect);
    void decreaseFramesTracked();
    void setMetaData(int _id, double _distance, const cv::String &_info);
    cv::Rect getRect(int _averagelast) const;

    int getFramesTracked() const;
    int getMetaId() const;
    cv::String getMetaInfo() const;
    double getMetaDistance() const;
    unsigned long getUuid() const;
    void setUuid(unsigned long value);
    bool getPosted2Srv() const;
    void setPosted2Srv(bool value);

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
 * @brief The MultyFaceTracker class
 */
class MultyFaceTracker
{
public:
    MultyFaceTracker();
    MultyFaceTracker(const cv::Ptr<cv::ofrt::FaceDetector> &_ptr, size_t _maxfaces);
    /**
     * @brief Search faces, then crop, resize and return them
     * @param Img - input image
     * @param size - output images size
     * @return vector of resized faces images
     */
    std::vector<cv::Mat> getResizedFaceImages(const cv::Mat &_img, const Size &_size, int _averagelast);
    /**
     * @brief getTrackedFaces get information about tracked faces
     * @return vector of tracked faces
     */
    std::vector<TrackedFace> getTrackedFaces() const;
    /**
     * @brief get pointer to particular TrackedFace object
     * @param i - id of the
     * @return pointer to underlying data
     */
    TrackedFace *at(size_t i);
    /**
     * @brief maxFaces
     * @return maximum of simultaneously tracked faces
     */
    size_t maxFaces() const;
    /**
     * @brief setFaceDetector - set face detection backend
     * @param _ptr - self explained
     * @param _maxfaces - maximum of simultaneously tracked faces
     */
    void setFaceDetector(const cv::Ptr<cv::ofrt::FaceDetector> &_ptr, size_t _maxfaces);

private:
    cv::Mat         __cropInsideFromCenterAndResize(const cv::Mat &input, const cv::Size &size);
    void            __enrollImage(const cv::Mat &_img);
    unsigned long   __nextUUID();

    cv::Ptr<cv::ofrt::FaceDetector> dPtr;    
    std::vector<TrackedFace> vtrackedfaces;

    unsigned long uuid;
};

}}

#endif // MULTYFACETRACKER_H
