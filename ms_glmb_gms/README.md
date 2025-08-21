IMPLEMENTATION
========================
- This Python implementation is ported from an implemented version of multi-sensor delta-GLMB.
- Sampling solutions (ranked assignments), `gibbs_multisensor_approx_cheap` is implemented in C++ based on __Algorithm 2: MM-Gibbs (Suboptimal)__ [0].  
- Adaptive birth is implemented based on [1] (implemented in C++ based on __Algorithm 1 Multi-sensor Adaptive Birth Gibbs Sampler__), Gaussian Likelihoods.

[0] Vo, B. N., Vo, B. T., & Beard, M. (2019). Multi-sensor multi-object tracking with the generalized labeled multi-Bernoulli filter. IEEE Transactions on Signal Processing, 67(23), 5952-5967.  
[1] Trezza, A., Bucci Jr, D. J., & Varshney, P. K. (2021). Multi-sensor Joint Adaptive Birth Sampler for Labeled Random Finite Set Tracking. arXiv preprint arXiv:2109.04355.  

USAGE
=====
- Install packages: `numpy`, C++ packages in cpp_gibbs (`python setup.py build develop`) `gibbs_multisensor_approx_cheap`, `sample_adaptive_birth`.  
- Install package: [`murty`](https://github.com/JohnPekl/murty) 
    - Clone project 
    - Run `python setup.py bdist_wheel` to create *.whl file e.g. `murty-1.0.0-cp311-cp311-win_amd64.whl`
    - Install *.whl file: `python -m pip install .\dist\murty-1.0.0-cp311-cp311-win_amd64.whl`
- Run: `python demo.py` or `audio_visual_sensor_fusion.py`
