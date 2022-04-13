#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Householder"
#include "eigen3/Eigen/QR"
#include "eigen3/Eigen/SVD"

#include "Timer.hpp"
#include "ConfigReader.hpp"
#include "FeatureTracker.hpp"
#include "FeatureMatcher.hpp"
#include "Multiview.hpp"
#include "ImgLoader.hpp"
#include "FeatureDetector.hpp"

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <memory>
#include <vector>
#include <numeric>
#include <filesystem>


/*
Goal of this file:
Assess the capacity of tracking vs matching in keeping long tracks.
**/


std::map<int, int> merge_map(std::map<int, int> prev_map_curr, std::map<int, int> curr_map_next){
    std::map<int, int> prev_map_next;
    for (auto &match : prev_map_curr){
        if (curr_map_next.count(match.second) != 0){
            prev_map_next[match.first] = curr_map_next[match.second];
        }
    }
    return prev_map_next;
}

std::map<int,int> map_itself(Frame f){
    std::map<int, int> map;
    for (auto & match : f.getMap()){
        map[match.first] = match.first;
    }
    return map;
}



int main(int argc, char** argv)
{

    if (argc < 2){
        std::cout << "Missing param file argument!" << std::endl;
        return -1;
    }
    // Load config
    std::string param_file = argv[1];
    std::cout << "parameter file full path: " << std::filesystem::current_path().string()+"/../"+param_file << std::endl;
    // Load config
    Config config = readParameterFile(std::filesystem::current_path().string()+"/../"+param_file);

    // initialize K
    Eigen::Matrix3f K;
    K << 458.654, 0, 367.215,
         0, 457.296, 248.375,
         0, 0, 1;
    
    // path and loader of the EUROC sequence
    std::vector<std::string> img_list = EUROC_img_loader(config.dataset_path);
    cv::Mat img_inc, img_last, img_origin;

    // Initialize the detector
    cv::Ptr<cv::FeatureDetector> detector = factoryDetector(config);

    Frame frame_origin, frame_last, frame_inc;
    std::map<int, int> origin_map_last, origin_map_inc, last_map_inc;

    // Stores results in .csv
    std::fstream results;
    std::string results_path = "results.csv";
    results.open(results_path, std::fstream::out);
    results << "nTracks\n";


    // Here we detect once and try to see how many tracks and matches follow
    // BEWARE you can't run both tracking and matching
    
    int counter = 0;
    std::string img_path;

    for (const auto & img_name : img_list){
        img_path = config.dataset_path + "/data/" + img_name;
        if (counter > config.max_nb_frames) break;

        // Origin init
        if (counter == 0){
            img_origin = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
            frame_origin.setImg(img_origin);
            if (config.enable_matcher){
                parallelDetectAndCompute(frame_last, detector, config.ncols, config.nrows);
            } else{
                parallelDetect(frame_last, detector, config.ncols, config.nrows);
            }            
            frame_last = frame_origin;
            origin_map_last = map_itself(frame_origin);

            results << origin_map_last.size() << "\n";
            counter ++;
            continue;
        }

        img_inc = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
        frame_inc.reset();
        frame_inc.setImg(img_inc);

        if (config.enable_tracker){
            last_map_inc.clear();
            int ntracked_features = track(frame_last, frame_inc, last_map_inc,
                                    config.tracker_width, config.tracker_height, config.nlevels_pyramids_klt,
                                    config.klt_max_err);


            if (ntracked_features < config.threshold_tracks){
                // Suboptimal: some kps are tracked twice
                last_map_inc.clear();
                frame_inc.reset();
                parallelDetect(frame_last, detector, config.nrows, config.ncols);
                ntracked_features = track(frame_last, frame_inc, last_map_inc,
                                    config.tracker_width, config.tracker_height, config.nlevels_pyramids_klt,
                                    config.klt_max_err);
                frame_origin = frame_last;
                origin_map_last = map_itself(frame_origin);
            }

            // Filtering with Essential Matrix
            cv::Mat cvMask;
            origin_map_inc = merge_map(origin_map_last, last_map_inc);
            std::vector<cv::Point2f> p2f_origin_track, p2f_inc_track;
            std::vector<int> all_the_ids;
            for (auto & match: origin_map_inc){
                p2f_origin_track.push_back(frame_origin.getKeyPointIdx(match.first)._cvKeyPoint.pt);
                p2f_inc_track.push_back(frame_inc.getKeyPointIdx(match.second)._cvKeyPoint.pt);
                all_the_ids.push_back(match.first);
            }
            if(!computeEssential(K, p2f_origin_track, p2f_inc_track, cvMask))break;
            
            // Removing the matches that are outliers 
            for (size_t k = 0; k<all_the_ids.size(); k++){
                if (cvMask.at<bool>(k) == 0){
                    origin_map_inc.erase(all_the_ids[k]);
                }
            }

            // Incoming is now last
            origin_map_last = origin_map_inc;

            results << cv::countNonZero(cvMask) << "\n";

        }

        if (config.enable_matcher){
            // DetectionAndCompute & storage for repopulation
            parallelDetectAndCompute(frame_inc, detector, config.nrows, config.ncols);
            last_map_inc.clear();
            int nmatched_features = match(frame_last, frame_inc, last_map_inc,
                                        config.matcher_width, config.matcher_height, config.threshold_matching);
        

            if (nmatched_features < config.threshold_tracks){
                break;
            }

            // Filtering with Essential Matrix
            cv::Mat cvMask;
            origin_map_inc = merge_map(origin_map_last, last_map_inc);
            std::vector<cv::Point2f> p2f_origin_match, p2f_inc_match;
            std::vector<int> all_the_ids;
            for (auto & match: origin_map_inc){
                p2f_origin_match.push_back(frame_origin.getKeyPointIdx(match.first)._cvKeyPoint.pt);
                p2f_inc_match.push_back(frame_inc.getKeyPointIdx(match.second)._cvKeyPoint.pt);
                all_the_ids.push_back(match.first);
            }
            if(!computeEssential(K, p2f_origin_match, p2f_inc_match, cvMask))break;

            // Removing the matches that are outliers 
            for (size_t k = 0; k<all_the_ids.size(); k++){
                if (cvMask.at<bool>(k) == 0){
                    origin_map_inc.erase(all_the_ids[k]);
                }
            }

            // Incoming is now Last
            origin_map_last = origin_map_inc;

            results << cv::countNonZero(cvMask) << "\n";
        }

        // Set inc as last (only for image this time)
        frame_last = frame_inc;
        counter++;
    }

    results.close();

    return 0;
}