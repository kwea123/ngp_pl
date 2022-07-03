# ngp_pl
Instant-ngp (only NeRF) in pytorch+cuda trained with pytorch-lightning (**high quality with high speed**). This repo aims at providing a concise pytorch interface to facilitate future research, and am grateful if you can share it (and a citation is highly appreciated)!

https://user-images.githubusercontent.com/11364490/177025079-cb92a399-2600-4e10-94e0-7cbe09f32a6f.mp4

https://user-images.githubusercontent.com/11364490/176821462-83078563-28e1-4563-8e7a-5613b505e54a.mp4

*  [Official CUDA implementation](https://github.com/NVlabs/instant-ngp/tree/master)
*  [torch-ngp](https://github.com/ashawkey/torch-ngp) another pytorch implementation that I highly referenced.

# :computer: Installation

This implementation has **strict** requirements due to dependencies on other libraries, if you encounter installation problem due to hardware/software mismatch, I'm afraid there is **no intention** to support different platforms (you are welcomed to contribute).

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

I compared the quality (average testing PSNR on `Synthetic-NeRF`) and the inference speed (on `Lego` scene) v.s. the concurrent work torch-ngp (default settings) and the paper, all trained for about 5 minutes:

| Method    | avg PSNR | FPS   | 
| :---:     | :---:    | :---: |
| torch-ngp | 31.46    | 7.8   |
| mine      | 32.38    | 36.2 |
| instant-ngp paper | **33.18** | **60** |

As for quality, mine is slightly better than torch-ngp, but the result might fluctuate across different runs.

As for speed, mine is faster than torch-ngp, but is still only half fast as instant-ngp. Speed is dependent on the scene (if most of the scene is empty, speed will be faster).

<p align="center">
  <img src="https://user-images.githubusercontent.com/11364490/176800109-38eb35f3-e145-4a09-8304-1795e3a4e8cd.png", width="45%">
  <img src="https://user-images.githubusercontent.com/11364490/176800106-fead794f-7e70-4459-b99e-82725fe6777e.png", width="45%">
  <br>
  <sup>Left: torch-ngp. Right: mine.</sup>
</p>

More details are in the following section.

# Benchmarks

To run benchmarks, use the scripts under `benchmarking`.

Followings are my results:

<details>
  <summary>Synthetic-NeRF</summary>

|       | Mic   | Ficus | Chair | Hotdog | Materials | Drums | Ship  | Lego  | AVG   |
| :---: | :---: | :---: | :---: | :---:  | :---:     | :---: | :---: | :---: | :---: |
| PSNR  | 35.23 | 33.64 | 34.78 | 36.76  | 28.77     | 25.61 | 29.57 | 34.69 | 32.38 |
| FPS   | 40.81 | 34.02 | 49.80 | 25.06  | 20.08     | 37.77 | 15.77 | 36.20 | 32.44 |

</details>

<details>
  <summary>Synthetic-NSVF</summary>

|       | Wineholder | Steamtrain | Toad | Robot | Bike | Palace | Spaceship | Lifestyle | AVG | 
| :---: | :---: | :---: | :---: | :---: | :---:  | :---:  | :---: | :---: | :---: |
| PSNR  | 31.06 | 35.65 | 34.49 | 36.23 | 36.99 | 36.36 | 35.48 | 33.96 | 35.03 |
| FPS   | 47.07 | 75.17 | 50.42 | 64.87 | 66.88 | 28.62 | 35.55 | 22.84 | 48.93 |

</details>

<details>
  <summary>Tanks and Temples</summary>

       | Ignatius | Truck | Barn | Caterpillar | Family | AVG   | 
|:---: | :---:    | :---: | :---: | :---:      | :---:  | :---: |
| PSNR |          |       |       |            |        |       |
| FPS  |          |       |       |            |        |       |

</details>

<details>
  <summary>BlendedMVS</summary>

       | Jade  | Fountain | Character | Statues | AVG   | 
|:---: | :---: | :---:    | :---:     | :---:   | :---: |
| PSNR |       |          |           |         |       |
| FPS  |       |          |           |         |       |

</details>


# TODO

- [ ] test multi-gpu training

- [ ] support custom dataset

- [ ] benchmark on public datasets