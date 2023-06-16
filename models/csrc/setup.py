import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]
# "helper_math.h" is copied from https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_math.h

sources = glob.glob('*.cpp')+glob.glob('*.cu')


setup(
    name='vren',
    version='2.0.1',
    author='kwea123, jnhwkim',
    author_email='kwea123@gmail.com, jnhwkim@gmail.com',
    description='cuda volume rendering library',
    long_description='cuda volume rendering library',
    ext_modules=[
        CUDAExtension(
            name='vren',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={'cxx': ['-O2'],
                                'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)