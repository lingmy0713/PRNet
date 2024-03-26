<div align="center">

# [TCI 2024] PRNet: Pyramid Restoration Network for RAW Image Super-Resolution

Mingyang Ling, Kan Chang, Mengyuan Huang, Hengxin Li, Shuping Dang, and Baoxin Li, 

[[`Paper`](https://ieeexplore.ieee.org/document/10463120)] [[`BibTeX`](#heart-citing-us)]

[![python](https://img.shields.io/badge/-Python_3.6_%7C_3.7-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

</div>

## 📌 Abstract

>Typically, image super-resolution (SR) methods are applied to the standard RGB (sRGB) images produced by the image signal processing (ISP) pipeline of digital cameras. However, due to error accumulation, low bit depth and the nonlinearity with scene radiance in sRGB images, performing SR on them is sub-optimal. To address this issue, a RAW image SR method called pyramid restoration network (PRNet) is proposed in this paper. Firstly, PRNet takes the low-resolution (LR) RAW image as input, and generates a rough estimation of the SR result in the linear color space. Afterwards, a pyramid refinement (PR) sub-network refines image details in the intermediate SR result and corrects its colors in a divide-and-conquer manner. To learn the appropriate colors for displaying, external guidance is extracted from the LR reference image in the sRGB color space, and then fed to the PR sub-network. To effectively incorporate the external guidance, the cross-layer correction module (CLCM), which fully investigates the long-range interactions between two input features, is introduced in the PR sub-network. Moreover, as different frequency components decomposed from the same image are highly correlated, in the PR sub-network, the refined features from a lower layer are utilized to support the feature refinement in an upper layer. Extensive experiments presented in this paper demonstrate that the proposed method is capable of recovering fine details and small structures in images while producing vivid colors that align with the output of a specific camera ISP pipeline.

## :book: Model


## :wrench: Dependencies

* Python == 3.7.13
* Pytorch == 1.13.0
* torchvision == 0.14.0
* numpy == 1.21.5
* skimage == 0.19.3

## :building_construction: Folder structure

1. Download the synthetic testing dataset (150 images) from [RawSR](https://github.com/xuxy09/RawSR).
[[Google Drive](https://drive.google.com/drive/folders/115ndMx97ZemzA_vV1Nf0W8gnEMrt32U5?usp=sharing)]
[[BaiduNetdisk(x2)](https://pan.baidu.com/s/1z972Ic5X3zmMdwkMeOwA2w)]
2. Download our [pretrained models](https://drive.google.com/file/d/14zHBnHAvu2Qkw3sINi8aHpaWtPzqCKfq/view?usp=sharing).
3. The following shows the basic folder structure.
```
├── Dataset # Please make a new folder for dataset.
│   ├── Blind_x2
│   │   └── TRAINING
│   │   └── TESTING
│   ├── Blind_x4 
│   │   └── TRAINING
│   │   └── TESTING
├── PRNet # code
│   ├── argument           # config files for code
│   ├── models           
│   ├── pretrained_models  # folder for pretrained models
│   ├── ...
```
   
## :rocket: Quick Test

Cd to './PRNet', run the following scripts to test models. 
```bash
    # Scale 2,4
    #-------------PRNet_x2 
    python main.py --n_patch_size 256  --dir_dataset ../Dataset/Blind_x2 --pre_train PRNet_x2.pth --b_test_only True

    #-------------PRNet_x4 
    python main.py --n_patch_size 128  --dir_dataset ../Dataset/Blind_x4 --pre_train PRNet_x4.pth --b_test_only True

```
If you want to save the output images for each dataset, you need to add `--b_save_results True` to test commands.

## :heart: Citing Us

If you find our repo useful for your research, please consider citing this paper and our [previous work](https://github.com/Hengxin-Li/TSCNN):

```
@ARTICLE{TSCNN,
  author={Chang, Kan and Li, Hengxin and Tan, Yufei and Ding, Pak Lun Kevin and Li, Baoxin},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={A Two-Stage Convolutional Neural Network for Joint Demosaicking and Super-Resolution}, 
  year={2022},
  volume={32},
  number={7},
  pages={4238-4254},
  doi={10.1109/TCSVT.2021.3129201}}
```
  
```
@ARTICLE{PRNet,
    author={Ling, Mingyang and Chang, Kan and Huang, Mengyuan and Li, Hengxin and Dang, Shuping and Li, Baoxin},
    journal={IEEE Transactions on Computational Imaging}, 
    title={PRNet: Pyramid Restoration Network for RAW Image Super-Resolution}, 
    year={2024},
    volume={10},
    pages={479-495},
    doi={10.1109/TCI.2024.3374084}}
}
```

## :handshake: Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch). 
We also refer to some other work such as [RawSR](https://github.com/xuxy09/RawSR), [RCAN](https://github.com/yulunzhang/RCAN), [pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter).
We thank these authors for sharing their codes.
