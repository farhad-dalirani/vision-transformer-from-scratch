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

### Training
After installing the project, you can start training using the `vit train` command. The training CLI is fully configuration-driven and allows you to override any field in the default experiment configs located in `src/vision_transformer/config/` via the `--set section.field=value` syntax. Multiple --set flags can be provided to customize the run without editing code. For example, the command below trains a Vision Transformer under the default configuration, with overrides for input image size (448), training duration (100 epochs), learning rate (3e-4), and Adam optimizer betas (0.9,0.98).

```python
vit train \
  --set model.image_size=448 \
  --set training.epochs=100 \
  --set optimizer.lr=3e-4 \
  --set optimizer.betas=0.9,0.98
```

Also, stop training can be resumed, just pass the path to saved checkpoint, for example:
```python
vit train \
  --set training.resume_path="./checkpoints/2026-02-07_21-53-15/last.pth"
```

## Results

## License
Released under the MIT License. For used datasets, please check their respective licenses.

## References
```
- Dosovitskiy, Alexey. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).
```
