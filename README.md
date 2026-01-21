# Vision Transformer (ViT) from Scratch

## Introduction
A clean PyTorch re-implementation of the Vision Transformer (ViT), introduced in “An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale” (ICLR 2021).

## Quick Installation

### 1. Create a virtual environment

``` bash
make venv
```

### 2. Activate it

``` bash
source .venv/bin/activate
```

### 3. Install Dependencies and CUDA-Enabled PyTorch

``` bash
make install-gpu
make install-dev
```
If you want to change the version of cuda enabled torch (currently CUDA 12.8), you can modify install-gpu section in `Makefile`.

### 4. Verify GPU support

``` bash
make check-gpu
```

## How To Use
Check supported commands and their options:

## Results

## License
Released under the MIT License. For used datasets, please check their respective licenses.

## References
```
- Dosovitskiy, Alexey. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).
```
