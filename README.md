# KLT vs Matching

A C++ code to compare the performances of feature tracking with LKT and descriptor matching implemented on OpenCV. An Image Loader for the Euroc dataset is implemented.

Compile and run:
---

Just type the following after editing the path of EU

```
mkdir build && cd build
cmake ..
make 
```

And then you can either run `./main` to evaluate performances on execution and tracking frame to frame, or `./main_sequence` to see tracking keyframe to frame (the results are reported in a .csv). You can edit the param.yaml file to put your own path and parameters

Warning:
---
* You can't enable both tracking and matching for `\main_sequence`
