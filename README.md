# OREPA: Online Convolutional Re-parameterization
This repo is the PyTorch implementation of our paper to appear in CVPR2022 on ["Online Convolutional Re-parameterization"](https://arxiv.org/abs/TODO), authored by
Mu Hu, Junyi Feng, Jiashen Hua, Baisheng Lai, Jianqiang Huang, [Xiaojin Gong](https://person.zju.edu.cn/en/gongxj) and [Xiansheng Hua](https://damo.alibaba.com/labs/city-brain) from Zhejiang University and Alibaba Cloud.

## What is Structural Re-parameterization?
+ Re-parameterization (Re-param) means different architectures can be mutually converted through equivalent transformation of parameters. For example, a branch of 1x1 convolution and a branch of 3x3 convolution, can be transferred into a single branch of 3x3 convolution for faster inference.
+ When the model for deployment is fixed, the task of re-param can be regarded as finding a complex training-time structure, which can be transfered back to the original one, for free performance improvements.

## Why do we propose Online RE-PAram? (OREPA)
+ While current re-param blocks ([ACNet](https://github.com/DingXiaoH/ACNet), [ExpandNet](https://github.com/GUOShuxuan/expandnets), [ACNetv2](https://github.com/DingXiaoH/DiverseBranchBlock), *etc*) are still feasible for small models, more complecated design for further performance gain on larger models could lead to unaffordable training budgets.
+ We observed that batch **normalization** (norm) layers are significant in re-param blocks, while their training-time non-linearity prevents us from optimizing computational costs during training.

## What is OREPA?
OREPA is a two-step pipeline.
+ Linearization: Replace the branch-wise norm layers to scaling layers to enable the linear squeezing of a multi-branch/layer topology.
+ Squeezing: Squeeze the linearized block into a single layer, where the convolution upon feature maps is reduced from multiple times to one.

## How does OREPA work?
+ Through OREPA we could reduce the training budgets while keeping a comparable performance. Then we improve accuracy by additional components, which brings minor extra training costs since they are merged in an online scheme.
+ The replacement of 

## ImageNet Results
+

Create a new issue for any code-related questions. Feel free to direct me as well at muhu@zju.edu.cn for any paper-related questions.

## Contents
1. [Dependency](#dependency)
2. [Checkpoints](#trained-models)
3. [Evaluation](#commands)
4. [](#commands)

3. [Commands](#commands)
4. [Citation](#citation)


## Dependency
Our released implementation is tested on.
+ TODO
+ Python 3.7.4 (Anaconda 2019.10)
+ PyTorch 1.3.1 / torchvision 0.4.2
+ NVIDIA CUDA 10.0.130
+ 4x NVIDIA V100 GPUs

```bash
pip install numpy matplotlib Pillow
pip install scikit-image
pip install opencv-contrib-python==3.4.2.17
```

## Trained Models
Download our pre-trained models with OREPA:
- [ResNet-18]()
- [ResNet-34]()
- [ResNet-50]()
- [ResNet-101]()
- [RepVGG-A0]()
- [RepVGG-A1]()
- [RepVGG-A2]()
- [WideResNet-18(x2)]()
- [ResNeXt-50]()
- [MobileNet-V1]()

 Note that we don't need to decompress the pre-trained models. Just load the files of .pth.tar format directly.

## Commands
A complete list of training options is available with
```bash
python main.py -h
```
### Training
![Training Pipeline](https://github.com/JUGGHM/PENet_ICRA2021/blob/main/images/Training.png "Training")

Here we adopt a multi-stage training strategy to train the backbone, DA-CSPN++, and the full model progressively. However, end-to-end training is feasible as well.

1. Train ENet (Part Ⅰ)
```bash
CUDA_VISIBLE_DEVICES="0,1" python main.py -b 6 -n e
# -b for batch size
# -n for network model
```


### Evalution
```bash
CUDA_VISIBLE_DEVICES="0" python main.py -b 1 -n e --evaluate [enet-checkpoint-path]
CUDA_VISIBLE_DEVICES="0" python main.py -b 1 -n pe --evaluate [penet-checkpoint-path]
# test the trained model on the val_selection_cropped data
```

### Transfer Learning on COCO and Cityscapes
```bash
CUDA_VISIBLE_DEVICES="0" python main.py -b 1 -n e --evaluate [enet-checkpoint-path]
CUDA_VISIBLE_DEVICES="0" python main.py -b 1 -n pe --evaluate [penet-checkpoint-path]
# test the trained model on the val_selection_cropped data
```



## Citation
If you use our code or method in your work, please cite the following:

	@inproceedings{hu22OREPA,
		title={Online Convolutional Re-parameterization},
		author={Mu Hu and Junyi Feng and Jiashen Hua and Baisheng Lai and Jianqiang Huang and Xiansheng Hua and Xiaojin Gong},
		booktitle={CVPR},
		year={2022}
	}

## Related Repositories
Codes of this work is developed upon Xiaohan Ding's re-param repositories ["Diverse Branch Block: Building a Convolution as an Inception-like Unit"](https://github.com/DingXiaoH/DiverseBranchBlock) and ["RepVGG: Making VGG-style ConvNets Great Again"](https://github.com/DingXiaoH/RepVGG) with similar protocols. [Xiaohan Ding](https://scholar.google.com/citations?user=CIjw0KoAAAAJ&hl=en) is a Ph.D. from Tsinghua University and an expert in structural re-parameterization.
