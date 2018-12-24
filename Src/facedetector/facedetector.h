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
    /**
     * @brief setPortions - change face region size
     * @param _xportion - portion of default width
     * @param _yportion - portion of default height
     */
    void setPortions(float _xportion, float _yportion);
    /**
     * @brief setShifts - change face region shift from center of detection
     * @param _xshift - shift of default horizontal position in portion of default width
     * @param _yshift - shift of default vertical position in portion of default height
     */
    void setShifts(float _xshift, float _yshift);

    float getXPortion() const;
    float getYPortion() const;
    float getXShift() const;
    float getYShift() const;

private:
    float xPortion;
    float yPortion;
    float xShift;
    float yShift;
};

}}

#endif // FACEDETECTOR_H
