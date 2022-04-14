#ifndef FEATUREMATCHER_H
#define FEATUREMATCHER_H

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <Frame.hpp>

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

std::vector<int> getKeypointsInBox(int search_width, int search_height, KeyPoint kp, Frame f){
    int x = kp._cvKeyPoint.pt.x;
    int y = kp._cvKeyPoint.pt.y;
    std::vector<int> indices_in_box;

    for (auto &keypoint : f.getKeyPointsVector()){
        if((keypoint._cvKeyPoint.pt.x < x - search_width/2) || (keypoint._cvKeyPoint.pt.x > x + search_width/2) 
            || (keypoint._cvKeyPoint.pt.y < y - search_height/2) || (keypoint._cvKeyPoint.pt.y > y + search_height/2))
            continue;
        indices_in_box.push_back(keypoint._id);
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

            // First we check if the octave of the pyramid is similar
            int octave_curr = kps_curr[index].octave;
            std::cout << octave_curr << " vs " <<  kp_prev.octave << std::endl;
            if (octave_curr <  kp_prev.octave-1 || octave_curr > kp_prev.octave){
                continue;
            }

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

int match(Frame f_prev, Frame f_curr, std::map<int, int> &prev_map_curr,
        int search_width, int search_height, float threshold){

    std::vector<int> indices_in_box;
    int number_of_matches = 0;

    for (auto & kp_prev: f_prev.getKeyPointsVector()){
        cv::Mat descriptor_prev = kp_prev._desc;

        indices_in_box = getKeypointsInBox(search_width, search_height, kp_prev, f_curr);

        // Case no feature in the surroundings
        if (indices_in_box.size() == 0) continue;

        // Here we take the best score, we can also do multi match
        int best_idx = 0;
        float best_score = threshold;
        for (auto & index : indices_in_box){

            // Check if the octaves are similar
            if (f_curr.getKeyPointIdx(index)._cvKeyPoint.octave <  kp_prev._cvKeyPoint.octave-1 
                || f_curr.getKeyPointIdx(index)._cvKeyPoint.octave > kp_prev._cvKeyPoint.octave){
                continue;
            }

            cv::Mat descriptor_curr = f_curr.getKeyPointIdx(index)._desc;
            float score = cv::norm(descriptor_prev, descriptor_curr, cv::NORM_HAMMING2);

            if (score < best_score){
                best_score = score;
                best_idx = index;
            }
        }

        if (best_score < threshold){
            prev_map_curr[kp_prev._id] = best_idx;
            number_of_matches++;
        }
    }
    return number_of_matches;
}

int match_bruteforce(std::vector<cv::KeyPoint> &kps_prev, std::vector<cv::KeyPoint> &kps_curr, 
        cv::Mat &descriptors_prev, cv::Mat &descriptors_curr,
        int search_width, int search_height, float threshold){

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    std::vector< std::vector<cv::DMatch> > knn_matches;
    matcher->knnMatch( descriptors_prev, descriptors_curr, knn_matches, 2 );
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.5f;
    int number_of_matches = 0;
    
    std::vector<cv::KeyPoint> kps_prev_new, kps_curr_new;
    cv::Mat descriptors_curr_new, descriptors_prev_new;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            kps_prev_new.push_back(kps_prev[knn_matches[i][0].queryIdx]);
            kps_curr_new.push_back(kps_curr[knn_matches[i][0].trainIdx]);
            descriptors_prev_new.push_back(descriptors_prev.row(knn_matches[i][0].queryIdx));
            descriptors_curr_new.push_back(descriptors_curr.row(knn_matches[i][0].trainIdx));
            number_of_matches++;
        }
    }

    kps_prev = kps_prev_new;
    kps_curr = kps_curr_new;
    descriptors_prev = descriptors_prev_new;
    descriptors_curr = descriptors_curr_new;
    return number_of_matches;
}

#endif