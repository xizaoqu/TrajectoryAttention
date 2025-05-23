
<br>
<p align="center">
<h1 align="center"><strong>Trajectory Attention For Fine-grained Video Motion Control</strong></h1>
  <p align="center"><span><a href="https://natanielruiz.github.io/"></a></span>
              <a href="https://github.com/xizaoqu">Zeqi Xiao<sup>1</sup></a	>
                <a href="https://vicky0522.github.io/Wenqi-Ouyang/">Wenqi Ouyang<sup>1</sup></a	>
              <a href="https://zhouyifan.net/about/">Yifan Zhou<sup>1</sup></a>
              <a href="https://williamyang1991.github.io/">Shuai Yang<sup>2</sup></a>
              <a href="https://scholar.google.com.hk/citations?user=jZH2IPYAAAAJ&hl=en/">Lei Yang<sup>3</sup></a>
              <a href="https://jianlou.github.io/">Jianlou Si<sup>3</sup></a>
              <a href="https://xingangpan.github.io/">Xingang Pan<sup>1</sup></a>    <br>
    <sup>1</sup>S-Lab, Nanyang Technological University, <br> <sup>2</sup>Wangxuan Institute of Computer Technology, Peking University,<br>  <sup>3</sup>Sensetime Research
    </p>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2411.19324" target='_blank'>
    <img src="https://img.shields.io/badge/arXiv-2411.19324-blue?">
  </a>
  <a href="https://xizaoqu.github.io/trajattn/" target='_blank'>
    <img src="https://img.shields.io/badge/Project-&#x1F680-blue">
  </a>
</p>

https://github.com/user-attachments/assets/c4a2243d-68d0-4993-ba33-14c2572de6a7


## 🏠 About
<!-- ![Teaser](assets/teaser.jpg) -->
<div style="text-align: center;">
    <img src="assets/teaser.jpg" alt="Dialogue_Teaser" width=100% >
</div>
Trajectory attention injects partial motion information by making content along trajectories consistent. It facilitates various tasks such as camera motion control on images and videos, and first-frame-guided video editing. Yellow boxes indicate reference contents. Green boxes indicate input frames. Blue boxes indicate output frames.

<div style="text-align: center;">
    <img src="assets/main.jpg" alt="Dialogue_Teaser" width=100% >
</div>
Our method allows for conditioning on trajectories from various sources -- 
such as camera motion derived from a single image, as shown in this figure. 
We inject these conditions into the model through trajectory attention, 
enabling explicit and fine-grained control over the motion in the generated video.


## Installation

1. Create a Conda Environment

This codebase is tested with the versions of PyTorch 2.1.0+cu121.

```
conda create -n trajattn python==3.10
conda activate trajattn
pip install -r requirements.txt
```

2. Download model weights
Download model weights from [huggingface](https://huggingface.co/zeqixiao/TrajectoryAttention).
```
wget https://huggingface.co/zeqixiao/TrajectoryAttention/resolve/main/trajattn_temp.pth?download=true
```

3. Clone Relevant Repositories and Download Checkpoints

```
# Clone the Depth-Anything-V2 repository
git clone https://github.com/DepthAnything/Depth-Anything-V2
# Download the Depth-Anything-V2-Large checkpoint
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true
# Overwrite the run.py
cp depth_anything/run.py Depth-Anything-V2/run.py
```

4. Save the checkpoints to the `checkpoints/` directory. You can also modify the checkpoint path in the running scripts if needed.

```
mkdir checkpoints
mv depth_anything_v2_vitl.pth\?download\=true checkpoints/depth_anything_v2_vitl.pth
mv trajattn_temp.pth\?download\=true checkpoints/trajattn_temp.pth
```

## Runnig

To control camera motion on images, execute the following script
```
sh image_control.sh
```

To control camera motion on videos, execute the following script
```
sh video_control.sh
```

To do video editing, execute the following script
```
sh video_editing.sh
```
## TODO

- [x] Release models and weight;
- [x] Release pipelines for single image camera motion control;
- [x] Release pipelines for video camera motion control;
- [x] Release pipelines for video editing;



## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@inproceedings{
xiao2025trajectory,
title={Trajectory attention for fine-grained video motion control},
author={Zeqi Xiao and Wenqi Ouyang and Yifan Zhou and Shuai Yang and Lei Yang and Jianlou Si and Xingang Pan},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=2z1HT5lw5M}
}
```

## 👏 Acknowledgements
- [SVD](https://github.com/Stability-AI/generative-models): Our model is tuned from SVD.
- [MiraData](https://github.com/mira-space/MiraData): We use the data collected by MiraData. 
- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2): We estimate depth map by Depth-Anything-V2.
- [Unimatch](https://github.com/autonomousvision/unimatch): We estimate optical flow map by Unimatch.
- [Cotracker](https://github.com/facebookresearch/co-tracker): We estimate point trajectories by Cotracker.
- [NVS_Solver](https://github.com/ZHU-Zhiyu/NVS_Solver): Our camera rendering code is based on NVS_Solver.
