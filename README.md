# KLT vs Matching

A C++ code to compare the performances of feature tracking with LKT and descriptor matching implemented on OpenCV. An Image Loader for the Euroc dataset is implemented.

Building the project:
---

```
mkdir build && cd build
cmake ..
make 
```

2 executables:  
- `./main params.yaml`: evaluate performances on execution and tracking frame to frame
- `./main_sequence params.yaml`: evaluate the persistance of tracks from frame to frame. Results in a .csv. Parameters in `params.yaml`

Scripts:
- `tracking_experiments.py`: run tracking and matching on EuRoc dataset

Warning:
---
* You can't enable both tracking and matching for `./main_sequence`
