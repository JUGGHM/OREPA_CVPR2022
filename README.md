# OREPA: Online Convolutional Re-parameterization
This repo is the PyTorch implementation of our paper to appear in CVPR2022 on ["Online Convolutional Re-parameterization"](https://arxiv.org/abs/TODO), authored by
Mu Hu, [Junyi Feng](https://github.com/Sixkplus), Jiashen Hua, Baisheng Lai, Jianqiang Huang, [Xiaojin Gong](https://person.zju.edu.cn/en/gongxj) and [Xiansheng Hua](https://damo.alibaba.com/labs/city-brain) from Zhejiang University and Alibaba Cloud.

## What is Structural Re-parameterization?
+ Re-parameterization (Re-param) means different architectures can be mutually converted through equivalent transformation of parameters. For example, a branch of 1x1 convolution and a branch of 3x3 convolution, can be transferred into a single branch of 3x3 convolution for faster inference.
+ When the model for deployment is fixed, the task of re-param can be regarded as finding a complex training-time structure, which can be transfered back to the original one, for free performance improvements.

<div align=center>
<img src="https://github.com/JUGGHM/OREPA_CVPR2022/blob/main/images/intro.png" width="600" />
</div>

## Why do we propose Online RE-PAram? (OREPA)
+ While current re-param blocks ([ACNet](https://github.com/DingXiaoH/ACNet), [ExpandNet](https://github.com/GUOShuxuan/expandnets), [ACNetv2](https://github.com/DingXiaoH/DiverseBranchBlock), *etc*) are still feasible for small models, more complecated design for further performance gain on larger models could lead to unaffordable training budgets.
+ We observed that batch normalization (norm) layers are significant in re-param blocks, while their training-time non-linearity prevents us from optimizing computational costs during training.

<img src="https://github.com/JUGGHM/OREPA_CVPR2022/blob/main/images/budgets.png" width="480"><img src="https://github.com/JUGGHM/OREPA_CVPR2022/blob/main/images/norm.PNG" width="330">

## What is OREPA?
OREPA is a two-step pipeline.
+ Linearization: Replace the branch-wise norm layers to scaling layers to enable the linear squeezing of a multi-branch/layer topology.
+ Squeezing: Squeeze the linearized block into a single layer, where the convolution upon feature maps is reduced from multiple times to one.

![Overview](https://github.com/JUGGHM/OREPA_CVPR2022/blob/main/images/overview.png)

## How does OREPA work?
+ Through OREPA we could reduce the training budgets while keeping a comparable performance. Then we improve accuracy by additional components, which brings minor extra training costs since they are merged in an online scheme.
+ We theoretically present that the removal of branch-wise norm layers risks a multi-branch structure degrading into a single-branch one, indicating that the norm-scaling layer replacement is critical for protecting branch diversity.

<img src="https://github.com/JUGGHM/OREPA_CVPR2022/blob/main/images/similarity.png" width="400"><img src="https://github.com/JUGGHM/OREPA_CVPR2022/blob/main/images/supp_grad.png" width="400">

## ImageNet Results
<div align=center>
<img src="https://github.com/JUGGHM/OREPA_CVPR2022/blob/main/images/imagenet1.PNG" width="450" />
</div>

![ImageNet2](https://github.com/JUGGHM/OREPA_CVPR2022/blob/main/images/imagenet2.PNG)

Create a new issue for any code-related questions. Feel free to direct me as well at muhu@zju.edu.cn for any paper-related questions.

## Contents
1. [Dependency](#dependency)
2. [Checkpoints](#checkpoints)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Transfer Learning on COCO and Cityscapes](#transfer-learning-on-coco-and-cityscapes)
6. [About Quantization and Gradient Tweaking](#about-quantization-and-gradient-tweaking)
7. [Citation](#citation)


## Dependency
Models released in this work is trained and tested on:
+ CentOS Linux
+ Python 3.8.8 (Anaconda 4.9.1)
+ PyTorch 1.9.0 / torchvision 0.10.0
+ NVIDIA CUDA 10.2
+ 4x NVIDIA V100 GPUs

```bash
pip install torch torchvision
pip install numpy matplotlib Pillow
pip install scikit-image
```

## Checkpoints
Download our pre-trained models with OREPA:
- [ResNet-18](https://drive.google.com/file/d/1Z0cfmxLLWD2xjgUXpE2Y_4vxS4ij16Sx/view?usp=sharing)
- [ResNet-34](https://drive.google.com/file/d/1tOC4yoslHF829Yb66_eE4qsFoazf6uoY/view?usp=sharing)
- [ResNet-50](https://drive.google.com/file/d/1Jn92wGHlFzkPjQxAmYdRU_EbSRQjsw0v/view?usp=sharing)
- [ResNet-101](https://drive.google.com/file/d/10L4fwYlB21vlMKOGIyXdlGBbf3l8nflI/view?usp=sharing)
- [RepVGG-A0](https://drive.google.com/file/d/1r674bxWKL5dwDA8OPEuGWN8zbiUGkqsx/view?usp=sharing)
- [RepVGG-A1](https://drive.google.com/file/d/1NVRn8Xave-0jY3R0xAXCShdBcML6f-3q/view?usp=sharing)
- [RepVGG-A2](https://drive.google.com/file/d/1ImHkTct0ACDtOw8sPgKhqerBFigg32s3/view?usp=sharing)
- [WideResNet-18(x2)](https://drive.google.com/file/d/1gseMlq9JGntggyZoK_Bjt7xiSPNs5deF/view?usp=sharing)
- [ResNeXt-50](https://drive.google.com/file/d/1p_eZDhMHQ_xmfXd8Z1RZnWfgwUbi7WH-/view?usp=sharing)

 Note that we don't need to decompress the pre-trained models. Just load the files of .pth.tar format directly.

## Training
A complete list of training options is available with
```bash
python train.py -h
python test.py -h
python convert.py -h
```

1. Train ResNets (ResNeXt and WideResNet included)
```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" python train.py -a ResNet-18 -t OREPA --data [imagenet-path]
# -a for architecture (ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-18-2x, ResNeXt-50)
# -t for re-param method (base, DBB, OREPA)
```

2. Train RepVGGs
```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" python train.py -a RepVGG-A0 -t OREPA_VGG --data [imagenet-path]
# -a for architecture (RepVGG-A0, RepVGG-A1, RepVGG-A2)
# -t for re-param method (base, RepVGG, OREPA_VGG)
```

## Evaluation
1. Use your self-trained model or our pretrained model
```bash
CUDA_VISIBLE_DEVICES="0" python test.py train [trained-model-path] -a ResNet-18 -t OREPA
```

2. Convert the training-time models into inference-time models
```bash
CUDA_VISIBLE_DEVICES="0" python convert.py [trained-model-path] [deploy-model-path-to-save] -a ResNet-18 -t OREPA
```

3. Evaluate with the converted model
```bash
CUDA_VISIBLE_DEVICES="0" python test.py deploy [deploy-model-path] -a ResNet-18 -t OREPA
```

## Transfer Learning on COCO and Cityscapes
We use [mmdetection](https://github.com/open-mmlab/mmdetection) and [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) tools on COCO and Cityscapes respectively. If you decide to use our pretrained model for downstream tasks, it is strongly suggested that the learning rate of the first stem layer should be fine adjusted, since the deep linear stem layer has a very different weight distribution from the vanilla one after ImageNet training. Contact [@Sixkplus](https://github.com/Sixkplus) (Junyi Feng) for more details on configurations and checkpoints of the reported ResNet-50-backbone models.

<div align=center>
<img src="https://github.com/JUGGHM/OREPA_CVPR2022/blob/main/images/coco_cs.PNG" width="400" />
</div>

## About Quantization and Gradient Tweaking
For re-param models, special weight regulization strategies are required for furthur quantization. Meanwhile, dynamic gradient tweaking or differential searching methods might greatly boost the performance. Currently we have not deployed such techniques to OREPA yet. However such methods could be probably applied to our industrial usage in the future. For experience exchanging and sharing on such topics please contact [@Sixkplus](https://github.com/Sixkplus) (Junyi Feng).


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
