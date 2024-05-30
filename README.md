# E3CM-Epipolar-Constrained-Cascaded-Correspondence-Matching


![截图 2024-05-30 13-35-31](https://github.com/bobojiang26/E3CM-Epipolar-Constrained-Cascaded-Correspondence-Matching/assets/91231457/cd3c79fa-df6a-40e0-9509-9ffc31b02e24)

<p align="center">
  <a href="https://arxiv.org/pdf/2308.16555">[Paper]</a>
</p>



# Abstract

Accurate and robust correspondence matching is of utmost importance for various 3D computer vision tasks. However, traditional explicit programming-based methods often struggle to handle challenging scenarios, and deep learning-based methods require large well-labeled datasets for network training. In this article, we introduce Epipolar-Constrained Cascade Correspondence (E3CM), a novel approach that addresses these limitations. Unlike traditional methods, E3CM leverages pre-trained convolutional neural networks to match correspondence, without requiring annotated data for any network training or fine-tuning. Our method utilizes epipolar constraints to guide the matching process and incorporates a cascade structure for progressive refinement of matches. We extensively evaluate the performance of E3CM through comprehensive experiments and demonstrate its superiority over existing methods.

# News
[2024-5-30] The code is released!

[2023-8-31] The paper is accepted by Nerocomputing!

# Installation
To set up the environment, run:

```sh
conda create -n e3cm

conda activate e3cm

pip install -r requirements.txt
```

# Test
We provide two images from the scene St. Peter's Basilica in the dataset Megadepth. You can test our model by running:

```
python3 test.py
```

Of course, you can try your own image pairs by changing the file address.


# Experiments

We also provide the evaluation code on HPatches and Megadepth.

For HPatches, you can run:
```
python3 eval_hpatches.py --data_root [your dataset root]
```

For Megadepth, you can run:
```
python3 eval_megadepth.py --data_root [your dataset root]
```


# BibTeX citation

If you use any ideas from the paper or code from this repo, please consider citing:

```
@article{zhou2023e3cm,
  title={E3CM: Epipolar-constrained cascade correspondence matching},
  author={Zhou, Chenbo and Su, Shuai and Chen, Qijun and Fan, Rui},
  journal={Neurocomputing},
  volume={559},
  pages={126788},
  year={2023},
  publisher={Elsevier}
}
```

