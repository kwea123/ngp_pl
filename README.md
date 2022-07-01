# ngp_pl
Instant-ngp in pytorch+cuda trained with pytorch-lightning (with only few lines of code and clear comments)

*  [Official CUDA implementation](https://github.com/NVlabs/instant-ngp/tree/master)
*  [torch-ngp](https://github.com/ashawkey/torch-ngp) another pytorch implementation that I highly referenced.

# :computer: Installation

This implementation has **strict** requirements due to dependencies on other libraries, if you encounter installation problem due to hardware/software mismatch, I'm afraid there is **no intention** to make the support for different platform currently.

## Hardware

* OS: Ubuntu 20.04
* NVIDIA GPU with Compute Compatibility >= 75 and memory > 8GB (Tested with RTX 2080 Ti), CUDA 11.3 (might work with older version)

## Software

* Clone this repo by `git clone https://github.com/kwea123/nerf_pl`
* Python>=3.8 (installation via [anaconda](https://www.anaconda.com/distribution/) is recommended, use `conda create -n ngp_pl python=3.8` to create a conda environment and activate it by `conda activate ngp_pl`)
* Python libraries
    * Install pytorch by `pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113`
    * Install `tinycudann` following their [instruction](https://github.com/NVlabs/tiny-cuda-nn#requirements) (compilation and pytorch extension)
    * Install `apex` following their [instruction](https://github.com/NVIDIA/apex#linux)
    * Install core requirements by `pip install -r requirements.txt`

* Cuda extension: Run `pip install models/csrc/ --use-feature=in-tree-build`

# :key: Training

A one line quickstart: `python train.py --root_dir <path/to/lego> --exp_name lego`

It will train the lego scene for 20k steps (each step with 8192 rays), and perform one testing at the end. The whole process should finish within about 5 minutes.

More options can be found in [opt.py](opt.py). Currently only nerf-synthetic dataset is supported.

# Comparison with torch-ngp

I compared the quality v.s. the concurrent work torch-ngp (default settings), both trained for about 5 minutes:

| test PSNR | mic   | ficus | chair | hotdog | materials | drums | ship  | lego  | AVG   |
| :---:     | :---: | :---: | :---: | :---:  | :---:     | :---: | :---: | :---: | :---: |
| torch-ngp | 34.48 | 30.57 | 32.16 | 36.21  | 28.17     | 24.04 | 31.18 | 34.88 | 31.46 |
| mine      | 35.00 | 33.51 | 34.40 | 36.60  | 28.91     | 25.37 | 30.27 | 34.64 | **32.34** |

mine is slightly better, but the result might fluctuate across different runs.

<p align="center">
  <img src="https://user-images.githubusercontent.com/11364490/176800109-38eb35f3-e145-4a09-8304-1795e3a4e8cd.png", width="45%">
  <img src="https://user-images.githubusercontent.com/11364490/176800106-fead794f-7e70-4459-b99e-82725fe6777e.png", width="45%">
  <br>
  <sup>Left: torch-ngp. Right: mine.</sup>
</p>

# TODO

[ ] multi-gpu training

[ ] other datasets

[ ] implement compact ray to accelerate inference
