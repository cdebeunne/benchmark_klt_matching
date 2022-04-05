#ifndef MULTIVIEW_H
#define MULTIVIEW_H

#include <opencv2/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "eigen3/Eigen/Core"

bool computeEssential(Eigen::Matrix3f K, std::vector<cv::Point2f> p2f_prev, std::vector<cv::Point2f> p2f_curr, cv::Mat &cvMask){

    if (p2f_prev.size() < 5) return false;
    float focal_length = K(0);
    cv::Point2f principal_pt(K(0,2), K(1,2));
    
    cv::findEssentialMat(p2f_prev, p2f_curr, focal_length, principal_pt, cv::RANSAC,
                        0.99, 1.0, cvMask);
    return true;
}

#endif