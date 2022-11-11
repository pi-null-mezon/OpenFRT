#include "facemark.h"

namespace cv { namespace ofrt {

Facemark::Facemark()
{
}

Facemark::~Facemark()
{
}

Rect Facemark::prepareRect(const cv::Rect &source, const cv::Rect &frame, float upscale)
{
    cv::Rect rect;
    if(source.width == source.height)
        rect = source;
    else if(source.width > source.height)
        rect = cv::Rect(source.x + (source.width - source.height) / 2, source.y, source.height, source.height);
    else
        rect = cv::Rect(source.x, source.y + (source.height - source.width) / 2, source.width, source.width);
    return (cv::Rect(rect.x - rect.width * (upscale - 1.0f) / 2.0f,
                     rect.y - rect.height * (upscale - 1.0f) / 2.0f,
                     rect.width*upscale,rect.height*upscale) & frame);
}

}}
