# Audio-Visual Sensor Fusion

This repository implements the **Audio-Visual Sensor Fusion Software Module**, responsible for **multi-object tracking (MOT)** of moving speakers by fusing detections from audio and video modalities.  
It is part of a modular audio-visual software architecture designed for reproducible indoor experiments with synthetic data.

## Audio-Visual Sensor Fusion Architecture

![Scene Diagram](Images/AVSensorFusionArchitecture.svg)

## Overview

The Sensor Fusion module follows the **Tracking-by-Detection (TBD)** paradigm and uses the **MS-GLMB (Multi-Scan Generalized Labeled Multi-Bernoulli)** algorithm to associate detections across time while handling:

- Measurement uncertainty
- Missed detections
- False alarms
- Object birth and death

The fusion of audio and video detections improves robustness compared to unimodal tracking, particularly in scenarios with occlusions or reverberant acoustic conditions.

## Features

- Multi-object tracking of moving speakers in 3D space
- Fusion of audio and video detections in a unified Bayesian framework
- Online processing with low computational complexity
- Robust handling of data association ambiguities
- Output compatible with TrackEval for quantitative evaluation

## Repository Structure

```
audiovisualsensorfusion/
├── ms_glmb_gms/   # directory (MS-GLMB)
│   ├── audio_visual_sensor_fusion.py   # Main fusion and tracking script
│   ├── gen_model.py                    # Motion model (linear kalman filter constant velocity)
│   ├── gen_truth.py                    # Load/Generate groundtruth tracks
│   ├── gen_meas.py                     # Load/Generate sensor measurements
│   ├── run_filter.py                   # MS-GLMB filter (predict, update)
│   ├── plot_results.py                 # 2D/3D track visualizations
│   ├── utils.py                        # Helper functions and data I/O
└── README.md
```

## Generated Scenario Folder Structure

Each simulation session follows a standardized directory structure shared across all software modules:

```
📁 experiment_001/
├── config.json
├── groundtruth_sources.jsonl
├── audio/
│   └── wav/
│       └── multichannel_audio_<timestamp>.wav
├── video/
│   ├── rgb/
│   │   └── RGB_frame_<timestamp>.png
│   └── depth/
│       └── Depth_frame_<timestamp>.png
├── localization/
│   ├── audio_localizations.jsonl
│   └── video_localizations.jsonl
└── tracking/
    ├── audio_tracking.jsonl
    ├── video_tracking.jsonl
    └── audio_video_tracking.jsonl
```

## Inputs and Outputs

### Inputs
- `audio_localizations.jsonl`  
  3D sound source detections produced by the Audio Detector module.
- `video_localizations.jsonl`  
  3D person detections produced by the Video Detector module.
- `groundtruth_sources.jsonl` *(optional)*  
  Ground-truth trajectories for evaluation.

### Output
- `audio_video_tracking_<Scenario>.jsonl`  
  Time-dependent 3D tracks of detected people/speakers, produced by the MS-GLMB tracker.

## Methodology

### Multi-Object Tracking with MS-GLMB

The MS-GLMB algorithm is used to track multiple speakers over time. It models the multi-speaker state as a **labeled Random Finite Set (RFS)**, enabling principled handling of:

- Track continuity
- Label consistency
- Data association across modalities

A constant-velocity motion model is employed, and Gaussian likelihoods are assumed for both audio and video measurements.


## Usage

1. Ensure audio and video localization files are available:
   - `audio_localizations.jsonl`
   - `video_localizations.jsonl`
2. Run the fusion script:

```bash
python audio_visual_sensor_fusion.py
```

3. The resulting people/speaker tracks are written to `audio_video_tracking_<Scenario>.jsonl`.

## Contact

Created by: **Laurens Sillekens**  
Part of a modular audio-visual sensor fusion framework for reproducible indoor experiments.

<br>
<br>
<br>
<br>
<br>

PYTHON LABELED RFS 
========================
This repository contains Python implementation of `jointglmb` GLMB [1] and `jointlmb` LMB [3]. The implementation are ported from `rfs_tracking_toolbox\jointlmb\gms` and `rfs_tracking_toolbox\jointglmb\gms` implemented in Matlab (it was done by Prof. Vo's research group). 
GLMB is originally, theoretically proposed in [0].

- A detail of how to implement Delta-GLMB (with two separated prediction and update steps) is given [2].   
- Adaptive birth is implemented based on [3] (Section __Adaptive Birth Distribution__), mainly focused on equation (75).  
- Sampling solutions (ranked assignments), `gibbs_multisensor_approx_cheap` is implemented in C++ based on __Algorithm 2: MM-Gibbs (Suboptimal)__ [4].  
- Adaptive birth is implemented based on [5] (implemented in C++ based on __Algorithm 1 Multi-sensor Adaptive Birth Gibbs Sampler__), Gaussian Likelihoods.

[0] Vo, Ba-Tuong, and Ba-Ngu Vo. "Labeled random finite sets and multi-object conjugate priors." IEEE Transactions on Signal Processing 61, no. 13 (2013): 3460-3475.  
[1] Vo, Ba-Ngu, Ba-Tuong Vo, and Hung Gia Hoang. "An efficient implementation of the generalized labeled multi-Bernoulli filter." IEEE Transactions on Signal Processing 65, no. 8 (2016): 1975-1987.  
[2] Vo, Ba-Ngu, Ba-Tuong Vo, and Dinh Phung. "Labeled random finite sets and the Bayes multi-target tracking filter." IEEE Transactions on Signal Processing 62, no. 24 (2014): 6554-6567.  
[3] Reuter, Stephan, Ba-Tuong Vo, Ba-Ngu Vo, and Klaus Dietmayer. "The labeled multi-Bernoulli filter." IEEE Transactions on Signal Processing 62, no. 12 (2014): 3246-3260.   
[4] Vo, B. N., Vo, B. T., & Beard, M. (2019). Multi-sensor multi-object tracking with the generalized labeled multi-Bernoulli filter. IEEE Transactions on Signal Processing, 67(23), 5952-5967.      
[5] Trezza, A., Bucci Jr, D. J., & Varshney, P. K. (2021). Multi-sensor Joint Adaptive Birth Sampler for Labeled Random Finite Set Tracking. arXiv preprint arXiv:2109.04355.
  
### Our Implementation for Visual GLMB in 2D/3D multi-object tracking
- Our implementation for 2D multi-object tracking with re-identification is released at [VisualRFS](https://github.com/linh-gist/VisualRFS)
- Our implementation for 3D multi-camera multi-object tracking with re-identification is released at [3D-Visual-MOT](https://github.com/linh-gist/3D-Visual-MOT)

USAGE
=====
GLMB
* Original Matlab source: `jointglmb_gms_matlab`
* Original Python porting: `jointglmb_gms_python`
* Improved version (code optimized, adaptive birth): `jointglmb_gms_python_fast`

LMB
* Original Matlab source: `jointlmb_gms_matlab`
* Original Python porting: `jointlmb_gms_python`
* Improved version (code optimized): `jointlmb_gms_python_fast`
* No adaptive birth is implemented for simplification (but can be implemented similar to jointglmb)

Gibbs Sampling
* Python package for an efficient algorithm for truncating the GLMB filtering density based on Gibbs sampling.
* The implementation is done in C++ and based on __Algorithm 1. Gibbs__ (and _"Algorithm 1a"_) of paper [1].
* Python wrapper for faster computation

MS-GLMB
* `gibbs_multisensor_approx_cheap` is implemented in C++.  
* Adaptive birth is implemented in C++, Gaussian Likelihoods.


### Contact
Linh Ma (linh.mavan@gm.gist.ac.kr), Machine Learning & Vision Laboratory, GIST, South Korea

### Citation
If you find this project useful in your research, please consider citing by:

```
@article{van2024visual,
      title={Visual Multi-Object Tracking with Re-Identification and Occlusion Handling using Labeled Random Finite Sets}, 
      author={Linh~Van~Ma and Tran~Thien~Dat~Nguyen and Changbeom~Shim and Du~Yong~Kim and Namkoo~Ha and Moongu~Jeon},
      journal={Pattern Recognition},
      volume = {156},
      year={2024},
      publisher={Elsevier}
}

@article{linh2024inffus,
      title={Track Initialization and Re-Identification for {3D} Multi-View Multi-Object Tracking}, 
      author={Linh Van Ma, Tran Thien Dat Nguyen, Ba-Ngu Vo, Hyunsung Jang, Moongu Jeon},
      journal={Information Fusion},
      volume = {111},
      year={2024},
      publisher={Elsevier}
}
```
