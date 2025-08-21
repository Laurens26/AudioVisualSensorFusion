# Gibbs Sampling

Python package for an efficient algorithm for truncating the GLMB filtering density based on Gibbs sampling.
 ## 1. Gibbs sampler
- `gibbs_jointpredupdt` initiates solution with output solution from `lapjv`. The implementation is done in C++ and based on __Algorithm 1. Gibbs__ (and _"Algorithm 1a"_) [1].  
- How to use? `gibbs_jointpredupdt(negative_log_cost_matrix, number_of_samples)`.  
## 2. MM-Gibbs (Suboptimal)
- Gibbs sampling based on minimallyMarkovian stationary distributions.  
- `gibbs_multisensor_approx_cheap` is implemented in C++ based on __Algorithm 2: MM-Gibbs (Suboptimal)__ [2], Gaussian Likelihoods.  
- How to use? `gibbs_multisensor_approx_cheap(death_cost, list_of_cost_matrix, number_of_samples)`. No negative logarithm for cost.  
## 3. Multi-sensor Adaptive Birth Gibbs Sampler
- `sample_adaptive_birth` is implemented in C++ based on __Algorithm 1 Multi-sensor Adaptive Birth Gibbs Sampler__ [3].
- How to use? (1) Create new object `AdaptiveBirth()`, (2) Initiate parameters `init_parameters(...)`, (3) `sample_adaptive_birth(list_of_unassigned_probability, list_of_measurement)`, output __mean, covariance, birth probability__ of new birth targets.

[1] Vo, Ba-Ngu, Ba-Tuong Vo, and Hung Gia Hoang. "An efficient implementation of the generalized labeled multi-Bernoulli filter." IEEE Transactions on Signal Processing 65, no. 8 (2016): 1975-1987.  
[2] Vo, B. N., Vo, B. T., & Beard, M. (2019). Multi-sensor multi-object tracking with the generalized labeled multi-Bernoulli filter. IEEE Transactions on Signal Processing, 67(23), 5952-5967.  
[3] Trezza, A., Bucci Jr, D. J., & Varshney, P. K. (2021). Multi-sensor Joint Adaptive Birth Sampler for Labeled Random Finite Set Tracking. arXiv preprint arXiv:2109.04355.  
## Requirements
General Requirements:
- Python 3.7 / 3.11.3
- C++ compiler (eg. Windows: Visual Studio 15 2017, Ubuntu: g++)

If Python 3.7 download release and copy/move content to /cpp_gibbs/pybind11/:
- https://github.com/pybind/pybind11/releases/tag/v2.10.4

If Python 3.11 download release and copy/move content to /cpp_gibbs/pybind11/:
- https://github.com/pybind/pybind11/releases/tag/v3.0.0
## Install

`python setup.py build develop`

Ouput will be a a compiled Python extension module file:
- \cpp_gibbs\lrfscpp.cp311-win_amd64.pyd

It is used in python e.g.:
- `import lrfscpp`

 or
- `from lrfscpp import gibbs_multisensor_approx_cheap, AdaptiveBirth`


Install Eigen for Windows (after the following steps, add include directory `C:\eigen-3.4.0` for example.)
1) Download Eigen 3.4.0 (NOT lower than this version) from it official website https://eigen.tuxfamily.org/ or [ZIP file here](https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip).
2) `mkdir build_dir`
3) `cd build_dir`
4) `cmake ../`
5) `make install`, this step does not require

Install Eigen for Linux
1) [install and use eigen3 on ubuntu 16.04](https://kezunlin.me/post/d97b21ee/) 
2) `sudo apt-get install libeigen3-dev` libeigen3-dev is installed install to `/usr/include/eigen3/` and `/usr/lib/cmake/eigen3`.
3) Thus, we must make a change in **CMakeLists.txt** `SET( EIGEN3_INCLUDE_DIR "/usr/local/include/eigen3" )` to `SET( EIGEN3_INCLUDE_DIR "/usr/include/eigen3/" )`.

LICENCE
=======
Linh Ma, Machine Learning & Vision Laboratory, GIST, South Korea