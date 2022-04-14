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
    

bool check_status(size_t j,
                  const std::vector<uchar>& status, 
                  const std::vector<float>& err,
                  const std::vector<uchar>& status_back,
                  const std::vector<float>& err_back, 
                  const Config& config) {
    if (status_back.size() == 0){
        return !status.at(j) || (err.at(j) > config.klt_max_err);
    }
    else{
        return (!status_back.at(j) ||  (err_back.at(j) > config.klt_max_err) 
              || !status.at(j)      ||  (err.at(j) > config.klt_max_err));
    }
}


int track(Frame &f_prev, Frame &f_curr, std::map<int, int> &prev_map_curr, Config config){
    
    // Create cv point list for tracking, we initialize optical flow with previous keypoints
    std::vector<cv::Point2f> p2f_prev = f_prev.getP2fVector();
    std::vector<cv::Point2f> p2f_curr = p2f_prev;
    cv::Mat img_curr = f_curr.getImg();
    std::vector<size_t> indices_prev = f_prev.getKeyPointIndices();

    // Configure and process KLT optical flow research
    std::vector<uchar> status;
    std::vector<float> err;
    std::vector<uchar> status_back;
    std::vector<float> err_back;
    // open cv default: 30 iters, 0.01 fpixels precision
    cv::TermCriteria crit(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);


    if (config.precompute_pyramids){
        ///////////////////////////////////////////////////////////////////////////////////
        // PRECOMPUTE PYRAMIDS
        ///////////////////////////////////////////////////////////////////////////////////
        // recover the precomputed one from previous frame
        std::vector<cv::Mat> pyr_prev = f_prev.getImgPyr();
        if (pyr_prev.size() == 0){
            cv::buildOpticalFlowPyramid(f_prev.getImg(), 
                                        pyr_prev, 
                                        {config.klt_patch_size, config.klt_patch_size}, 
                                        config.nlevels_pyramids_klt,
                                        config.pyr_with_derivatives);
            f_prev.setImgPyr(pyr_prev);    
        }
        // compute a new one
        std::vector<cv::Mat> pyr_curr;
        cv::buildOpticalFlowPyramid	(img_curr, 
                                    pyr_curr, 
                                    {config.klt_patch_size, config.klt_patch_size}, 
                                    config.nlevels_pyramids_klt,
                                    config.pyr_with_derivatives);
        f_curr.setImgPyr(pyr_curr);

        cv::calcOpticalFlowPyrLK(
                pyr_prev,
                pyr_curr, 
                p2f_prev,
                p2f_curr,
                status, err,
                {config.klt_patch_size, config.klt_patch_size}, 
                config.nlevels_pyramids_klt,
                crit,
                (cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS));
        
        if (config.klt_use_backward){
            // Process the other way: current->previous
            cv::calcOpticalFlowPyrLK(
                    pyr_curr,
                    pyr_prev, 
                    p2f_curr,
                    p2f_prev,
                    status_back, err_back,
                    {config.klt_patch_size, config.klt_patch_size}, 
                    config.nlevels_pyramids_klt,
                    crit,
                    (cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS));
        }
    }

    else {
        ///////////////////////////////////////////////////////////////////////////////////
        // PYRAMIDS COMPUTED MULTIPLE TIMES
        ///////////////////////////////////////////////////////////////////////////////////
        cv::Mat img_prev = f_prev.getImg();

        // Process one way: previous->current with current init with previous
        cv::calcOpticalFlowPyrLK(
                img_prev,
                img_curr, 
                p2f_prev,
                p2f_curr,
                status, err,
                {config.klt_patch_size, config.klt_patch_size}, 
                config.nlevels_pyramids_klt,
                crit,
                (cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS));
        
        if (config.klt_use_backward){
            // Process the other way: current->previous
            cv::calcOpticalFlowPyrLK(
                    img_curr,
                    img_prev, 
                    p2f_curr,
                    p2f_prev,
                    status_back, err_back,
                    {config.klt_patch_size, config.klt_patch_size}, 
                    config.nlevels_pyramids_klt,
                    crit,
                    (cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS));
        }
    }



    // Delete point if KLT failed or if point is outside the image
    int valid_matches = 0;
    for (size_t j = 0; j < status.size(); j++) {

        if(check_status(j, status, err, status_back, err_back, config)) {
            // !!! removing keypoints double the computation time
            // f_prev.removeKeyPointIdx(indices_prev.at(j));
            continue;
        }

        // We keep the initial point and add the tracked point
        KeyPoint kp;
        kp._id = j;
        kp._cvKeyPoint = cv::KeyPoint(p2f_curr.at(j), 1);
        f_curr.addKeyPoint(kp);

        // Update the map
        prev_map_curr[indices_prev.at(j)] = j;

        valid_matches++;
        // Other checks? Distance between track points?
    }

    return valid_matches;
}

#endif