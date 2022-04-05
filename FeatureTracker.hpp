#ifndef FEATURETRACKER_H
#define FEATURETRACKER_H

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

#include <Frame.hpp>
#include <iostream>
#include <vector>

int track(cv::Mat img_prev, cv::Mat img_curr, std::vector<cv::KeyPoint> &kps_prev, std::vector<cv::KeyPoint> &kps_curr,
           int search_width = 21, int search_height = 21, int pyramid_level = 3, float klt_max_err = 50.){
    
    // Create cv point list for tracking, we initialize optical flow with previous keypoints
    std::vector<cv::Point2f> p2f_curr, p2f_prev;
    for (auto & keypoint : kps_prev){
        p2f_prev.push_back(keypoint.pt);
        p2f_curr.push_back(keypoint.pt);
    }

    // Configure and process KLT optical flow research
    std::vector<uchar> status;
    std::vector<float> err;
    // open cv default: 30 iters, 0.01 fpixels precision
    cv::TermCriteria crit(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);

    // Process one way: previous->current with current init with previous
    cv::calcOpticalFlowPyrLK(
            img_prev,
            img_curr, 
            p2f_prev,
            p2f_curr,
            status, err,
            {search_width, search_height}, 
            pyramid_level,
            crit,
            (cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS));
    
    // Delete point if KLT failed or if point is outside the image
    cv::Size img_size = img_prev.size();
    std::vector<cv::Point2f> p2f_curr_new, p2f_prev_new;

    for (size_t i = 0; i < status.size(); i++) {
        // Invalid match if one of the OF failed or KLT error is too high
        if(!status.at(i) || (err.at(i) > klt_max_err)) {
            continue;
        }

        // Check if tracked points in the second image are in the image
        if( (p2f_curr.at(i).x < 0) ||
            (p2f_curr.at(i).y < 0) ||
            (p2f_curr.at(i).x > img_size.width) ||
            (p2f_curr.at(i).y > img_size.height) ) {
            continue;
        }
        p2f_curr_new.push_back(p2f_curr.at(i));
        p2f_prev_new.push_back(p2f_prev.at(i));
    }
    
    // Process the other way: current->previous
    std::vector<uchar> status_back;
    std::vector<float> err_back;
    cv::calcOpticalFlowPyrLK(
            img_curr,
            img_prev, 
            p2f_curr_new,
            p2f_prev_new,
            status_back, err_back,
            {search_width, search_height}, 
            pyramid_level,
            crit,
            (cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS));

    // Delete point if KLT failed or if point is outside the image
    int valid_matches = 0;
    int i_back = -1;
    std::vector<cv::KeyPoint> kps_prev_new, kps_curr_new;
    for (size_t i = 0; i < status.size(); i++) {
        
        // Invalid match if one of the OF failed or KLT error is too high
        if(!status.at(i)  ||  (err.at(i) > klt_max_err) ) {
            continue;
        }

        // Check if tracked points in the second image are in the image
        if( (p2f_curr.at(i).x < 0) ||
            (p2f_curr.at(i).y < 0) ||
            (p2f_curr.at(i).x > img_size.width) ||
            (p2f_curr.at(i).y > img_size.height) ) {
            continue;
        }

        i_back ++;

        // Go into back flow indices
        if(!status_back.at(i_back)  ||  (err_back.at(i_back) > klt_max_err) ) {
            continue;
        }

        // Check if tracked points in the second image are in the image
        if( (p2f_prev_new.at(i_back).x < 0) ||
            (p2f_prev_new.at(i_back).y < 0) ||
            (p2f_prev_new.at(i_back).x > img_size.width) ||
            (p2f_prev_new.at(i_back).y > img_size.height) ) {
            continue;
        }

        // We keep the initial point and add the tracked point
        kps_prev_new.push_back(kps_prev.at(i));
        kps_curr_new.push_back(cv::KeyPoint(p2f_curr_new.at(i_back),1));

        valid_matches++;
        // Other checks? Distance between track points?
    }
    kps_curr = kps_curr_new;
    kps_prev = kps_prev_new;

    return valid_matches;
}

int track(Frame f_prev, Frame f_curr,
          int search_width = 21, int search_height = 21, int pyramid_level = 3, float klt_max_err = 50.){
    
    // Create cv point list for tracking, we initialize optical flow with previous keypoints
    std::vector<cv::Point2f> p2f_curr, p2f_prev;
    p2f_curr = f_curr.getP2fVector();
    p2f_prev = f_prev.getP2fVector();

    // Configure and process KLT optical flow research
    std::vector<uchar> status;
    std::vector<float> err;
    // open cv default: 30 iters, 0.01 fpixels precision
    cv::TermCriteria crit(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);

    // Process one way: previous->current with current init with previous
    cv::calcOpticalFlowPyrLK(
            img_prev,
            img_curr, 
            p2f_prev,
            p2f_curr,
            status, err,
            {search_width, search_height}, 
            pyramid_level,
            crit,
            (cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS));
    
    // Delete point if KLT failed or if point is outside the image
    cv::Size img_size = img_prev.size();
    std::vector<cv::Point2f> p2f_curr_new, p2f_prev_new;

    for (size_t i = 0; i < status.size(); i++) {
        // Invalid match if one of the OF failed or KLT error is too high
        if(!status.at(i) || (err.at(i) > klt_max_err)) {
            continue;
        }

        // Check if tracked points in the second image are in the image
        if( (p2f_curr.at(i).x < 0) ||
            (p2f_curr.at(i).y < 0) ||
            (p2f_curr.at(i).x > img_size.width) ||
            (p2f_curr.at(i).y > img_size.height) ) {
            continue;
        }
        p2f_curr_new.push_back(p2f_curr.at(i));
        p2f_prev_new.push_back(p2f_prev.at(i));
    }
    
    // Process the other way: current->previous
    std::vector<uchar> status_back;
    std::vector<float> err_back;
    cv::calcOpticalFlowPyrLK(
            img_curr,
            img_prev, 
            p2f_curr_new,
            p2f_prev_new,
            status_back, err_back,
            {search_width, search_height}, 
            pyramid_level,
            crit,
            (cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS));

    // Delete point if KLT failed or if point is outside the image
    int valid_matches = 0;
    int i_back = -1;
    std::vector<cv::KeyPoint> kps_prev_new, kps_curr_new;
    for (size_t i = 0; i < status.size(); i++) {
        
        // Invalid match if one of the OF failed or KLT error is too high
        if(!status.at(i)  ||  (err.at(i) > klt_max_err) ) {
            continue;
        }

        // Check if tracked points in the second image are in the image
        if( (p2f_curr.at(i).x < 0) ||
            (p2f_curr.at(i).y < 0) ||
            (p2f_curr.at(i).x > img_size.width) ||
            (p2f_curr.at(i).y > img_size.height) ) {
            continue;
        }

        i_back ++;

        // Go into back flow indices
        if(!status_back.at(i_back)  ||  (err_back.at(i_back) > klt_max_err) ) {
            continue;
        }

        // Check if tracked points in the second image are in the image
        if( (p2f_prev_new.at(i_back).x < 0) ||
            (p2f_prev_new.at(i_back).y < 0) ||
            (p2f_prev_new.at(i_back).x > img_size.width) ||
            (p2f_prev_new.at(i_back).y > img_size.height) ) {
            continue;
        }

        // We keep the initial point and add the tracked point
        kps_prev_new.push_back(kps_prev.at(i));
        kps_curr_new.push_back(cv::KeyPoint(p2f_curr_new.at(i_back),1));

        valid_matches++;
        // Other checks? Distance between track points?
    }
    kps_curr = kps_curr_new;
    kps_prev = kps_prev_new;

    return valid_matches;
}

#endif