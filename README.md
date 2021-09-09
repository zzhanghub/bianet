<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://arxiv.org/abs/2004.14582">
    <img src="imgs/BiANet_logo.png" alt="Logo" width="210" height="100">
  </a>

  <h3 align="center">Bilateral Attention Network for RGB-D Salient Object Detection</h3>
  <p align="center">
    Published in IEEE Transactions on Image Processing (TIP)
    <br />
    <a href="https://ieeexplore.ieee.org/document/9321705?source=authoralert"><strong> [Paper 📄]</strong></a>
    <a href="https://arxiv.org/abs/2004.14582"><strong> [ArXiv 🌐]</strong></a>
    <!-- <a href="https://arxiv.org/abs/2004.14582"><strong> [中译版 🐼]</strong></a> -->
    <a href="http://zhaozhang.net"><strong> [Homepage 🏠] »</strong></a>
    <br />
  </p>
</p>

***
<!-- The official repo of the paper
[Bilateral Attention Network for RGB-D Salient Object Detection](https://ieeexplore.ieee.org/document/9321705?source=authoralert) -->


## Prerequisites
#### Environments
* PyTorch >= 1.0
* Ubuntu 18.04


<!-- USAGE EXAMPLES -->
## Usage
1. Download the [model parameters](#download) and [datasets](http://dpfan.net/d3netbenchmark/)
2. Configure  `test.sh`

``` 
--backbones vgg16+vgg11+res50+res2_50 (Multiple items are connected with '+')
--datasets dataset1+dataset2+dataset3
--param_root param (pretrained model path)
--input_root your_data_root (categorize by subfolders)
--save_root your_output_root
```

3. Run by
```
sh test.sh
```
## <span id="download">Model parameters and prediction results <span>
|   | Model parameters |  Prediction results |
|  ----  | ----  | ----  |
|  **VGG-16**  | [[Google Drive]](https://drive.google.com/file/d/1yfE2-4GH-QJo5JvvJbKRwXgzaRQ5e8h_/view?usp=sharing) [[Baidu Pan (bfrn)]](https://pan.baidu.com/s/1gXkDYUU0wxzM2EjyBoO6Yg) | [[Google Drive]](https://drive.google.com/file/d/1JoDT4O2gtoaiafWky1T8nNcPxbGGeaz8/view?usp=sharing) [[Baidu Pan (8ft8)]](https://pan.baidu.com/s/1J0JgFqqUCW-CM5EbkDUgyw) |
|  VGG-11  | [[Google Drive]](https://drive.google.com/file/d/1TdTvZmPIbPfaX_BYI7dNTUoMI7IVXvFe/view?usp=sharing) [[Baidu Pan (2a5c)]](https://pan.baidu.com/s/1Usr-SNCPZADyISaIXPEZxA) | [[Google Drive]](https://drive.google.com/file/d/14aP1634QFjc0wQu8Unjme0lsmaJtlnFp/view?usp=sharing) [[Baidu Pan (d0t7)]](https://pan.baidu.com/s/1U-7hkmvfN8Pjj0pnC8VLGQ) |
|  ResNet-50  | [[Google Drive]](https://drive.google.com/file/d/13vHFAR44v2bojEJppoB058QV0Vc9-Tm7/view?usp=sharing) [[Baidu Pan (o9l2)]](https://pan.baidu.com/s/1m0p7IN4GB2BWCcoj6kM_lw) | [[Google Drive]](https://drive.google.com/file/d/1E9kNo8PIZ4tbeoluQ0cJctFRKtffM_IY/view?usp=sharing) [[Baidu Pan (ssgc)]](https://pan.baidu.com/s/1cAnx70hZ5tLWEMmZXRlvVQ) |
|  Res2Net-50  | [[Google Drive]](https://drive.google.com/file/d/1DppyXLs_toFi6bM5ZbGWip35BxLGfw4y/view?usp=sharing) [[Baidu Pan (k761)]](https://pan.baidu.com/s/1ycs9SI5bmIKBUbcNsrR7qQ) | [[Google Drive]](https://drive.google.com/file/d/1at-K6DfKNP2Gnao9f0v9agmzADkgt0Ik/view?usp=sharing) [[Baidu Pan (h3t9)]](https://pan.baidu.com/s/1YHVrDEl1-dCHgS2Fuc1Qzw) |

## Citation
```
@article{zhang2020bianet,
  title={Bilateral attention network for rgb-d salient object detection},
  author={Zhang, Zhao and Lin, Zheng and Xu, Jun and Jin, Wenda and Lu, Shao-Ping and Fan, Deng-Ping},
  journal={IEEE Transactions on Image Processing (TIP)},
  volume={30},
  pages={1949-1961},
  doi={10.1109/TIP.2021.3049959},
  year={2021},
}
```

## Contact
If you have any questions, feel free to contact me via `zzhang🥳mail😲nankai😲edu😲cn`
