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
        Frame(cv::Mat cvimg): _cvimg(cvimg) {};

        std::vector<cv::KeyPoint> getKeyPointsVector() const;
        std::vector<cv::Point2f> getP2fVector() const;
        cv::Mat getCvImage() const;
        cv::Mat getDescriptors() const;
        void addKeyPoint(KeyPoint kp);
        void removeKeyPoint(KeyPoint kp);

    private:

    cv::Mat _cvimg;
    std::unordered_map<int, KeyPoint> _mapkps;


};

#endif