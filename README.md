# SCNet
The official PyTorch implementation of CVPR 2020 paper ["Improving Convolutional Networks with Self-Calibrated Convolutions"](http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf)

## Introduction
we present a novel self-calibrated convolution that explicitly expands fields-of-view of each convolutional layer
through internal communications and hence enriches the
output features. In particular, unlike the standard convolutions that fuse spatial and channel-wise information using
small kernels (e.g., 3 × 3), our self-calibrated convolution
adaptively builds long-range spatial and inter-channel dependencies around each spatial location through a novel
self-calibration operation. Thus, it can help CNNs generate
more discriminative representations by explicitly incorporating richer information. Our self-calibrated convolution
design is simple and generic, and can be easily applied to
augment standard convolutional layers without introducing
extra parameters and complexity. Extensive experiments
demonstrate that when applying our self-calibrated convolution into different backbones, the baseline models can be
significantly improved in a variety of vision tasks, including image recognition, object detection, instance segmentation, and keypoint detection, with no need to change network architectures.
<div align="center">
  <img src="https://github.com/backseason/SCNet/blob/master/figures/SC-Conv.png">
</div>
<p align="center">
  Figure 1: Diagram of self-calibrated convolution.
</p>

## Useage
### Requirement
PyTorch>=0.4.1
### Examples 
```
git clone https://github.com/backseason/SCNet.git

from scnet import scnet50
model = scnet50(pretrained=True)

```
Input image should be normalized as follows:
```
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
```

(The pretrained model should be downloaded automatically by default.
You may also choose to download them manually by the links listed below.)

## Pretrained models
| model |#Params | MAdds | FLOPs |top-1 error| top-5 error | Link 1 | Link 2 |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| SCNet-50  | 25.56M | 4.0G | 7.9G  | 22.19 | 6.08 |[GoogleDrive](https://drive.google.com/open?id=1rA266TftaUymbtPTVHCJYoxDwl6K4gLr) | [BaiduYun](https://pan.baidu.com/s/13js74yBkCsGAFx6N8ki7UA) password: **95p5**
| SCNet-101 | 44.57M | 7.2G | 14.4G | 21.06 | 5.75 |[GoogleDrive](https://drive.google.com/open?id=11-rW7l9vl-HGrOoCktEjRBPxMeKw334x) | [BaiduYun](https://pan.baidu.com/s/1qtwTxKbhzdxYqADsbgCcpQ) password: **38oh**

## Applications
Other applications such as Classification, Instance segmentation, Object detection, Semantic segmentation, and Human keypoint detection can be found on https://mmcheng.net/scconv/.

## Citation
If you find this work or code is helpful in your research, please cite:
```
@inproceedings{liu2020scnet,
 title={Improving Convolutional Networks with Self-Calibrated Convolutions},
 author={Jiang-Jiang Liu and Qibin Hou and Ming-Ming Cheng and Changhu Wang and Jiashi Feng},
 booktitle={IEEE CVPR},
 year={2020},
}
```
## Contact
If you have any questions, feel free to contact me via: `j04.liu(at)gmail.com`.
