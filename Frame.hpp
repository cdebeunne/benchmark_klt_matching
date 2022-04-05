#ifndef FRAME_H
#define FRAME_H

#include <opencv2/core.hpp>

#include <iostream>
#include <unordered_map>

struct KeyPoint
{
    int _id;
    cv::KeyPoint _cvKeyPoint;
    cv::Mat _desc;
};

class Frame
{
    public:
        Frame() {_maxidx = 0;};
        Frame(cv::Mat cvimg): _cvimg(cvimg) {_maxidx = 0;};

        std::vector<KeyPoint> getKeyPointsVector() const;
        std::vector<cv::KeyPoint> getCvKeyPointsVector() const;
        std::vector<cv::Point2f> getP2fVector() const;
        std::vector<size_t> getKeyPointIndices() const;
        std::unordered_map<size_t, KeyPoint> getMap() const {return _mapkps;};
        KeyPoint getKeyPointIdx(size_t idx) const;
        cv::Mat getCvImage() const;
        cv::Mat getDescriptors() const;

        void addKeyPoint(KeyPoint kp);
        void addKeyPoint(cv::KeyPoint cvkp);
        void addKeyPoint(cv::KeyPoint cvkp, cv::Mat desc);
        void removeKeyPoint(KeyPoint kp);
        void removeKeyPointIdx(size_t idx);
        void setImg(cv::Mat cvimg);
        void reset();

    private:

    cv::Mat _cvimg;
    std::unordered_map<size_t, KeyPoint> _mapkps;
    size_t _maxidx;
};

#endif