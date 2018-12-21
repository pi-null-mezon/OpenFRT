#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

#include <opencv2/core.hpp>

namespace cv { namespace ofrt {

/**
 * @brief Base abstarct for all face detectors
 */
class FaceDetector
{
public:
    FaceDetector();
    virtual ~FaceDetector();

    /**
     * @brief detectFaces - should return bounding boxes for all faces detected on image
     * @param _img - input image
     * @return vector of biunding boxes
     */
    virtual std::vector<Rect> detectFaces(InputArray &_img) const = 0;
};

}}

#endif // FACEDETECTOR_H
