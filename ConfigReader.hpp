#ifndef CONFIGREADER_H
#define CONFIGREADER_H

#include <yaml-cpp/yaml.h>

// This structure contains the configuration parameters located in the config file.
struct Config {
    std::string dataset_path;

    std::string detector;
    int threshold_fast;
    int npoints;
    float scale_factor;
    int nlevels_pyramids;

    int tracker_width;
    int tracker_height;
    int nlevels_pyramids_klt;
    float klt_max_err;

    int matcher_width;
    int matcher_height;
    float threshold_matching;

    bool debug;
    bool enable_tracker;
    bool enable_matcher;
    int nimages;
};

Config readParameterFile(const std::string path){
    YAML::Node yaml_file = YAML::LoadFile(path);
    Config config;

    // Script mode
    config.dataset_path = yaml_file["dataset_path"].as<std::string>();
    config.debug = yaml_file["debug"].as<bool>();
    config.enable_matcher = yaml_file["enable_matcher"].as<bool>();
    config.enable_tracker = yaml_file["enable_tracker"].as<bool>();
    config.nimages = yaml_file["nimages"].as<int>();

    // Config detector
    config.detector = yaml_file["detector"].as<std::string>();
    config.threshold_fast = yaml_file["threshold_fast"].as<int>();
    config.npoints = yaml_file["npoints"].as<int>();
    config.scale_factor = yaml_file["scale_factor"].as<float>();
    config.nlevels_pyramids = yaml_file["nlevels_pyramids"].as<int>();

    // Config tracker
    config.tracker_width = yaml_file["tracker_width"].as<int>();
    config.tracker_height = yaml_file["tracker_height"].as<int>();
    config.nlevels_pyramids_klt = yaml_file["nlevels_pyramids_klt"].as<int>();
    config.klt_max_err = yaml_file["klt_max_err"].as<float>();

    // Config matcher
    config.matcher_width = yaml_file["matcher_width"].as<int>();
    config.matcher_height = yaml_file["matcher_height"].as<int>();
    config.threshold_matching = yaml_file["threshold_matching"].as<float>();

    return config;
}

#endif