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
    TrackedFace(int _historylength=TARGET_VIDEO_FPS); // controls how long face tracker should not be dropped when face disappears

    void clearMetadata();
    void updatePosition(const cv::Rect &_brect);
    void decreaseFramesTracked();
    void setMetaData(int _id, double _distance, const cv::String &_info);
    cv::Rect getRect(int _averagelast=1) const;

    int getFramesTracked() const;
    int getMetaId() const;
    cv::String getMetaInfo() const;
    double getMetaDistance() const;
    unsigned long getUuid() const;
    void setUuid(unsigned long value);
    bool getPosted2Srv() const;
    void setPosted2Srv(bool value);
    int getUnknownInRow() const;

private:

    std::vector<cv::Rect>   vhistoryrects;  // history of face positions
    int                     historylength;  // length of history
    int                     pos;            // current position in vhistoryrects
    bool                    posted2Srv;     // posted to identification server
    int                     framesTracked;  // how long face is tracked
    cv::String              metaInfo;
    int                     metaId;
    int                     unknownInRow;   // stores how many times metaId has been set to -1
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
     * @brief Detect faces and update tracking information
     * @param _img - input image
     */
    void enrollImage(const cv::Mat &_img);
    /**
     * @brief Search faces, then crop, resize and return them
     * @param _img - input image
     * @param _size - output images size
     * @return vector of resized faces images along with positional id of corresponding tracked face object
     */
    std::vector<std::pair<size_t,cv::Mat>> getResizedFaceImages(const cv::Mat &_img, const Size &_size, int _averagelast);
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
    /**
     * @brief clear all tracked faces data
     */
    void clear();


    static cv::Mat  __cropInsideFromCenterAndResize(const cv::Mat &input, const cv::Size &size);

private:
    unsigned long   __nextUUID();

    cv::Ptr<cv::ofrt::FaceDetector> dPtr;    
    std::vector<TrackedFace> vtrackedfaces;

    unsigned long uuid;
};

}}

#endif // MULTYFACETRACKER_H
