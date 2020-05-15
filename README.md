# SCNet
The official PyTorch implementation of CVPR 2020 paper ["Improving Convolutional Networks with Self-Calibrated Convolutions"](http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf)

## Update
- 2020.5.15 
  - Pretrained model of SCNet-50_v1d with more than 2% improvement on ImageNet top1 acc (80.47 v.s. 77.81). compared with original version of SCNet-50 is released! 
  - **SCNet-50_v1d achieves comparable performance on other applications such as object detection and instance segmentation to our original SCNet-101 version.**
  - Because of limited GPU resources, the pretrained model of SCNet-101_v1d will be released later, as well as more applications' results.

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
| SCNet-50  | 25.56M | 4.0G | 7.9G  | 22.19 | 6.08 |[GoogleDrive](https://drive.google.com/open?id=1rA266TftaUymbtPTVHCJYoxDwl6K4gLr) | [BaiduYun](https://pan.baidu.com/s/13js74yBkCsGAFx6N8ki7UA) pwd: **95p5**
| **SCNet-50_v1d**  | 25.58M | 4.7G | 9.4G  | 19.53 | 4.68 |[GoogleDrive](https://drive.google.com/open?id=1EWZ4vELJVFNry6SRJEza5-T9nKuoZWgv) | [BaiduYun](https://pan.baidu.com/s/17dUFIXfTaXBgv3UJTFqJZg) pwd: **hmmt**
| SCNet-101 | 44.57M | 7.2G | 14.4G | 21.06 | 5.75 |[GoogleDrive](https://drive.google.com/open?id=11-rW7l9vl-HGrOoCktEjRBPxMeKw334x) | [BaiduYun](https://pan.baidu.com/s/1qtwTxKbhzdxYqADsbgCcpQ) pwd: **38oh**

## Applications (more coming soon...)
### Object detection
We use Faster R-CNN architecture with feature pyramid networks (FPNs) as baselines. We adopt the widely used [mmdetection](https://github.com/open-mmlab/mmdetection) framework to run all our experiments. Performances are reported on the COCO minival set.
| backbone | AP | AP.5 | AP.75 | APs | APm | APl |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| ResNet-50 |	37.6 | 59.4 | 40.4 | 21.9 | 41.2 | 48.4 |
| SCNet-50 | 40.8 | 62.7 | 44.5 | 24.4 | 44.8 | 53.1 | 
| **SCNet-50_v1d** | 41.8 | 62.9 | 45.5 | 24.8 | 45.3 | 54.8 | 
| ResNet-101 | 39.9 | 61.2 | 43.5 | 23.5 | 43.9 | 51.7 | 
| SCNet-101 | 42.0 | 63.7 | 45.5 | 24.4 | 46.3 | 54.6 | 

### Instance segmentation
We use Mask R-CNN architecture with feature pyramid networks (FPNs) as baselines. We adopt the widely used [mmdetection](https://github.com/open-mmlab/mmdetection) framework to run all our experiments. Performances are reported on the COCO minival set.
| backbone | AP | AP.5 | AP.75 | APs | APm | APl |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
 | esNet-50 | 35.0 | 56.5 | 37.4 | 18.3 | 38.2 | 48.3 |  
 | SCNet-50 | 37.2 | 59.9 | 39.5 | 17.8 | 40.3 | 54.2 | 
 | **SCNet-50_v1d** | 38.5 | 60.6 | 41.3 | 20.8 | 42.0 | 52.6 | 
 | ResNet-101 | 36.7 | 58.6 | 39.3 | 19.3 | 40.3 | 50.9 |  
 | SCNet-101 | 38.4 | 61.0 | 41.0 | 18.2 | 41.6 | 56.6 | 

Other applications such as Instance segmentation, Object detection, Semantic segmentation, and Human keypoint detection can be found on https://mmcheng.net/scconv/.

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
