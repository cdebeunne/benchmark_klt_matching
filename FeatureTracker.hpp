#ifndef FEATURETRACKER_H
#define FEATURETRACKER_H

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

#include <iostream>
#include <vector>

int track(cv::Mat img_prev, cv::Mat img_curr, std::vector<cv::Point2f> &kps_prev, std::vector<cv::Point2f> &kps_curr_flow,
           int search_width = 21, int search_height = 21, int pyramid_level = 3, float klt_max_err = 50.){

    // Configure and process KLT optical flow research
    std::vector<uchar> status;
    std::vector<float> err;
    // open cv default: 30 iters, 0.01 fpixels precision
    cv::TermCriteria crit(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);

    // Process one way: previous->current with current init with previous
    cv::calcOpticalFlowPyrLK(
            img_prev,
            img_curr, 
            kps_prev,
            kps_curr_flow,
            status, err,
            {search_width, search_height}, 
            pyramid_level,
            crit,
            (cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS));
    
    // Delete point if KLT failed or if point is outside the image
    cv::Size img_size = img_prev.size();
    std::vector<cv::Point2f> kps_curr_new, kps_prev_new;

    for (size_t i = 0; i < status.size(); i++) {
        // Invalid match if one of the OF failed or KLT error is too high
        if(!status.at(i) || (err.at(i) > klt_max_err)) {
            continue;
        }

        // Check if tracked points in the second image are in the image
        if( (kps_curr_flow.at(i).x < 0) ||
            (kps_curr_flow.at(i).y < 0) ||
            (kps_curr_flow.at(i).x > img_size.width) ||
            (kps_curr_flow.at(i).y > img_size.height) ) {
            continue;
        }
        kps_curr_new.push_back(kps_curr_flow.at(i));
        kps_prev_new.push_back(kps_prev.at(i));
    }
    
    // Process the other way: current->previous
    status.clear();
    err.clear();
    cv::calcOpticalFlowPyrLK(
            img_curr,
            img_prev, 
            kps_curr_new,
            kps_prev_new,
            status, err,
            {search_width, search_height}, 
            pyramid_level,
            crit,
            (cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS));

    // Delete point if KLT failed or if point is outside the image
    int valid_matches = 0;
    kps_curr_flow.clear();
    kps_prev.clear();
    for (size_t i = 0; i < status.size(); i++) {

        // Invalid match if one of the OF failed or KLT error is too high
        if(!status.at(i)  ||  (err.at(i) > klt_max_err) ) {
            continue;
        }

        // Check if tracked points in the second image are in the image
        if( (kps_prev_new.at(i).x < 0) ||
            (kps_prev_new.at(i).y < 0) ||
            (kps_prev_new.at(i).x > img_size.width) ||
            (kps_prev_new.at(i).y > img_size.height) ) {
            continue;
        }
        kps_curr_flow.push_back(kps_curr_new.at(i));
        kps_prev.push_back(kps_prev_new.at(i));

        valid_matches++;
        // Other checks? Distance between track points?
    }

    return valid_matches;
}

#endif