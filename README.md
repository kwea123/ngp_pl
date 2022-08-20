# ngp_pl

[MGTV competition](https://challenge.ai.mgtv.com/contest/detail/15) 9th place solution. 

Design details are in [design.pdf](design.pdf)

# :computer: Installation

## Hardware

* OS: Ubuntu 20.04
* NVIDIA GPU with Compute Compatibility >= 75 and memory > 6GB (Tested with RTX 2080 Ti), CUDA 11.3 (might work with older version)
* 32GB RAM (in order to load full size images)

## Software

### Matting

* Install `PaddleSeg` following [here](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/Matting)
* Download [modnet-hrnet_w18.pdparams](https://paddleseg.bj.bcebos.com/matting/models/modnet-hrnet_w18.pdparams) and put under `PaddleSeg` root directory.
* Copy `matting.py` in this repo to `PaddleSeg/Matting` directoy. 

### Reconstruction

* Clone this repo by `git clone https://github.com/kwea123/ngp_pl`
* Python>=3.8 (installation via [anaconda](https://www.anaconda.com/distribution/) is recommended, use `conda create -n ngp_pl python=3.8` to create a conda environment and activate it by `conda activate ngp_pl`)
* Python libraries
    * Install pytorch by `pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113`
    * Install `torch-scatter` following their [instruction](https://github.com/rusty1s/pytorch_scatter#installation)
    * Install `tinycudann` following their [instruction](https://github.com/NVlabs/tiny-cuda-nn#requirements) (compilation and pytorch extension)
    * Install `apex` following their [instruction](https://github.com/NVIDIA/apex#linux)
    * Install core requirements by `pip install -r requirements.txt`

* Cuda extension: Upgrade `pip` to >= 22.1 and run `pip install models/csrc/` (please run this each time you `pull` the code)

# :key: Training

## Matting

Change the variable `root_dir` in `PaddleSeg/Matting/matting.py` to your `test_b` path.

From `PaddleSeg/Matting`, run

```python3
python matting.py
```

It will generate matted images into `test_b_` folder under the same folder as `test_b`.

The total time taken on a RTX 2080 Ti is about 1.5h and the VRAM is around 5GB.

## Reconstruction

Change the variable `root_dir` in `train_all.py` to the folder containing `test_b`.

From this repo, run

```python3
python train_all.py
```

It will train the reconstruction for all scenes.

The total time taken on a RTX 2080 Ti is about 12hr and the VRAM is around 10GB.

# :mag_right: Testing

Change the variable `root_dir` in `eval_all.py` to the folder containing `test_b`.

Change the variable `rect_text_path` in `eval_all.py` to the rectangle text path.

From this repo, run

```python3
python eval_all.py
```

It will generate the (cropped) novel views under `results/mgtv_test_b` for all scenes.

The total time taken on a RTX 2080 Ti is about 40m and the VRAM is around 2GB.
