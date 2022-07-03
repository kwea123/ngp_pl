# ngp_pl
Instant-ngp (only NeRF) in pytorch+cuda trained with pytorch-lightning (**high quality with high speed**). I hope this repo can facilitate future research, and am grateful if you can share it (and a citation is highly appreciated)!

https://user-images.githubusercontent.com/11364490/177025079-cb92a399-2600-4e10-94e0-7cbe09f32a6f.mp4

https://user-images.githubusercontent.com/11364490/176821462-83078563-28e1-4563-8e7a-5613b505e54a.mp4

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
    * Install pytorch (1.11.0) by `pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113`
    * Install `tinycudann` following their [instruction](https://github.com/NVlabs/tiny-cuda-nn#requirements) (compilation and pytorch extension)
    * Install `apex` following their [instruction](https://github.com/NVIDIA/apex#linux)
    * Install core requirements by `pip install -r requirements.txt`

* Cuda extension: Run `pip install models/csrc/ --use-feature=in-tree-build`

# :books: Data preparation

Download preprocessed datasets from [NSVF](https://github.com/facebookresearch/NSVF#dataset).

# :key: Training

Quickstart: `python train.py --root_dir <path/to/lego> --exp_name Lego`

It will train the lego scene for 20k steps (each step with 8192 rays), and perform one testing at the end. The training process should finish within about 5 minutes (saving testing image is slow, add `--no_save_test` to disable). Testing PSNR will be shown at the end.

More options can be found in [opt.py](opt.py).

# Comparison with torch-ngp and the paper

I compared the quality (testing PSNR) and the inference speed (on `Lego` scene) v.s. the concurrent work torch-ngp (default settings) and the paper, all trained for about 5 minutes:

|    | split | Mic   | Ficus | Chair | Hotdog | Materials | Drums | Ship  | Lego  | AVG   | FPS | 
| :---:     | :---: | :---: | :---: | :---: | :---:  | :---:     | :---: | :---: | :---: | :---: | :---: |
| torch-ngp | train | 34.48 | 30.57 | 32.16 | 36.21 | 28.17 | 24.04 | 31.18 | 34.88 | 31.46 | 7.8 |
| mine | train | 35.00 | 33.51 | 34.40 | 36.60 | 28.91 | 25.37 | 30.27 | 34.98 | **32.32** | **31** |
| instant-ngp paper | all? | 36.22 | 33.51 | 35.00 | 37.40 | 29.78 | 26.02 | 31.10 | 36.39 | 33.18 | 60 |
| *mine | trainval | 36.30 | 34.75 | 35.34 | 37.86 | 29.90 | 26.37 | 31.16 | 35.86 | **33.44** | 31 |

As for quality, mine is slightly better than torch-ngp, but the result might fluctuate across different runs. Using `trainval` set, mine almost matches the paper.

As for speed, mine is faster than torch-ngp, but is still only half fast as instant-ngp. Speed is dependent on the scene (if most of the scene is empty, speed will be faster).

*: used with `hard_sampling` to train more on difficult rays (1-2min slower).

<p align="center">
  <img src="https://user-images.githubusercontent.com/11364490/176800109-38eb35f3-e145-4a09-8304-1795e3a4e8cd.png", width="45%">
  <img src="https://user-images.githubusercontent.com/11364490/176800106-fead794f-7e70-4459-b99e-82725fe6777e.png", width="45%">
  <br>
  <sup>Left: torch-ngp. Right: mine.</sup>
</p>

# TODO

- [ ] test multi-gpu training

- [ ] support custom dataset

- [ ] benchmark on public datasets