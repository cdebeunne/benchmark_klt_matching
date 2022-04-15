#ifndef CONFIGREADER_H
#define CONFIGREADER_H

#include <yaml-cpp/yaml.h>
#include <iostream>

// This structure contains the configuration parameters located in the config file.
struct Config {
    std::string dataset_path;

    std::string detector;
    int nrows;
    int ncols;
    int threshold_fast;
    int nb_orb_detected;
    float scale_factor;
    int nlevels_pyramids;

    int klt_patch_size;
    int nlevels_pyramids_klt;
    float klt_max_err;
    bool precompute_pyramids;
    bool pyr_with_derivatives;
    bool klt_use_backward;

    int matcher_width;
    int matcher_height;
    float threshold_matching;

    bool debug;
    bool enable_tracker;
    bool enable_matcher;
    int max_nb_frames;
    int threshold_tracks;

    int nb_cells_h;
    int nb_cells_v;
};

Config readParameterFile(const std::string path){
    YAML::Node yaml_file = YAML::LoadFile(path);
    Config config;

    // Script mode
    config.dataset_path = yaml_file["dataset_path"].as<std::string>();
    config.debug = yaml_file["debug"].as<bool>();
    config.enable_matcher = yaml_file["enable_matcher"].as<bool>();
    config.enable_tracker = yaml_file["enable_tracker"].as<bool>();
    config.max_nb_frames = yaml_file["max_nb_frames"].as<int>();
    config.threshold_tracks = yaml_file["threshold_tracks"].as<int>();

    // Config detector
    config.nrows = yaml_file["nrows"].as<int>();
    config.ncols = yaml_file["ncols"].as<int>();
    config.detector = yaml_file["detector"].as<std::string>();
    config.threshold_fast = yaml_file["threshold_fast"].as<int>();
    config.nb_orb_detected = yaml_file["nb_orb_detected"].as<int>();
    config.scale_factor = yaml_file["scale_factor"].as<float>();
    config.nlevels_pyramids = yaml_file["nlevels_pyramids"].as<int>();

    // Config tracker
    config.klt_patch_size = yaml_file["klt_patch_size"].as<int>();
    config.klt_max_err = yaml_file["klt_max_err"].as<float>();
    config.nlevels_pyramids_klt = yaml_file["nlevels_pyramids_klt"].as<int>();
    config.precompute_pyramids = yaml_file["precompute_pyramids"].as<bool>();
    config.pyr_with_derivatives = yaml_file["pyr_with_derivatives"].as<bool>();
    config.klt_use_backward = yaml_file["klt_use_backward"].as<bool>();

    // Config matcher
    config.matcher_width = yaml_file["matcher_width"].as<int>();
    config.matcher_height = yaml_file["matcher_height"].as<int>();
    config.threshold_matching = yaml_file["threshold_matching"].as<float>();

    // Experiment detect in cell grid
    config.nb_cells_h = yaml_file["nb_cells_h"].as<float>();
    config.nb_cells_v = yaml_file["nb_cells_v"].as<float>();
    
    return config;
}

#endif