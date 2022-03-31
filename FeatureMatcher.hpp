#ifndef FEATUREMATCHER_H
#define FEATUREMATCHER_H

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>
#include <vector>

std::vector<int> getKeypointsInBox(int search_width, int search_height, cv::KeyPoint kp, std::vector<cv::KeyPoint> kps){
    int x = kp.pt.x;
    int y = kp.pt.y;
    std::vector<int> indices_in_box;

    for (size_t i = 0; i < kps.size(); i++){
        cv::KeyPoint keypoint = kps.at(i);
        if((keypoint.pt.x < x - search_width/2) || (keypoint.pt.x > x + search_width/2) 
            || (keypoint.pt.y < y - search_height/2) || (keypoint.pt.y > y + search_height/2))
            continue;
        indices_in_box.push_back(i);
    }

    return indices_in_box;
}

int match(std::vector<cv::KeyPoint> &kps_prev, std::vector<cv::KeyPoint> &kps_curr, 
        cv::Mat &descriptors_prev, cv::Mat &descriptors_curr,
        int search_width, int search_height, float threshold){

    std::vector<int> indices_in_box;
    int number_of_matches = 0;
    std::vector<cv::KeyPoint> kps_prev_new, kps_curr_new;
    cv::Mat desc_prev_new, desc_curr_new;

    for (size_t i=0; i < kps_prev.size(); i++){
        cv::KeyPoint kp_prev = kps_prev.at(i);
        cv::Mat descriptor_prev = descriptors_prev.row(i);

        indices_in_box = getKeypointsInBox(search_width, search_height, kp_prev, kps_curr);

        // Case no feature in the surroundings
        if (indices_in_box.size() == 0) continue;

        // Here we take the best score, we can also do multi match
        int best_idx = 0;
        float best_score = threshold;
        for (auto & index : indices_in_box){
            cv::Mat descriptor_curr = descriptors_curr.row(index);
            float score = cv::norm(descriptor_prev, descriptor_curr, cv::NORM_HAMMING2);

            if (score < best_score){
                best_score = score;
                best_idx = index;
            }
        }

        if (best_score < threshold){
            kps_prev_new.push_back(kp_prev);
            kps_curr_new.push_back(kps_curr.at(best_idx));
            desc_prev_new.push_back(descriptor_prev);
            desc_curr_new.push_back(descriptors_curr.row(best_idx));
            number_of_matches++;
        }
    }

    kps_prev = kps_prev_new;
    kps_curr = kps_curr_new;
    descriptors_prev = desc_prev_new;
    descriptors_curr = desc_curr_new;
    return number_of_matches;
}

#endif