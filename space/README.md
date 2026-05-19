---
title: MIRO
emoji: 🎨
colorFrom: red
colorTo: pink
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
hardware: zero-a10g
suggested_hardware: zero-a10g
short_description: Multi-reward conditioned text-to-image diffusion (ICML 2026)
models:
  - nicolas-dufour/miro
  - nicolas-dufour/miro-ablations
tags:
  - text-to-image
  - diffusion
  - flow-matching
  - miro
license: mit
---

# MIRO — Multi-reward conditioned text-to-image

Interactive demo for **MIRO** (ICML 2026): drag the seven reward sliders to
steer generation toward each axis (CLIP alignment, aesthetic quality,
ImageReward, PickScore, HPSv2, VQAScore, SciScore).

- 📄 [Paper](https://arxiv.org/abs/2510.25897)
- 🌐 [Project page](https://nicolas-dufour.github.io/miro/)
- 💻 [Code](https://github.com/nicolas-dufour/miro)
- 🤗 [Models](https://huggingface.co/nicolas-dufour/miro)
- 🐍 [PyPI](https://pypi.org/project/miro-t2i/) — `pip install miro-t2i`

The demo runs on **ZeroGPU H200**: the GPU is allocated on-demand per
inference call. First request takes ~10s of cold-start; subsequent ones are
typically <5s for 4 samples at 50 steps.
