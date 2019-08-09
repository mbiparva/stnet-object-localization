# STNet: Selective Tuning of Convolutional Networks for Object Localization
By Mahdi Biparva (PhD Student @ York University, Toronto)

This repository contains the ___STNet___ implementation for _weakly-supervised object localization_.

You can check the research webpage for further information.

[STNet for object localization](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w40/Biparva_STNet_Selective_Tuning_ICCV_2017_paper.pdf "STNet: Selective Tuning of Convolutional Networks for Object Localization") is presented at [_Mutual Benefits Of Cognitive And Computer Vision_](https://sites.google.com/site/mbcc2017w/home "Mutual Benefits Of Cognitive And Computer Vision") workshop in ICCV 2017.

### License

StNet for Object Localization is released under the The GNU GPL v3.0 License [see LICENSE for details].

### Citing StNet for Object Localization

If you find StNet for Object Localization useful in your research, please consider citing the research paper:

    @InProceedings{Biparva_2017_STNet,
    author = {Biparva, Mahdi and Tsotsos, John},
    title = {STNet: Selective Tuning of Convolutional Networks for Object Localization},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV) Workshops},
    month = {Oct},
    year = {2017}
    }

## Contents

1. [Introduction](#introduction)
2. [Requirements: Software](#requirements-software)
3. [Requirements: Hardware](#requirements-hardware)
4. [Installation](#installation)
5. [Preparation](#preparation)
6. [Demo: Localization](#demo-weakly-supervised-object-localization)
7. [Demo: Visualization](#demo-class-hypothesis-visualization)
8. [Future Work](#future-work)

## Introduction

STNet is a computational visual attention model originated from _Selective Tuning_ model. It is aimed to implement a selective Top-Down visual processing pass along with the current Bottom-up feed forward and back propagation passes. Selective attention is a well-established and fundamental phenomenon in human vision. STNet is an attempt to understand and implement an inspired mechanism to deep learning models. Below we show the overall architecure of the network. Please consult the paper for further details.

![](http://drive.google.com/uc?export=view&id=1ohHzOmdRHuBQa034RVpti_Nf_wM4etpk)


## Requirements: Software

STNet is implemented in two languages: C/CUDA an Python. It is currently integrated and compatible with [PyTorch](https://github.com/pytorch/pytorch) neural network library. We leverage the [custom C extension API](https://pytorch.org/docs/stable/notes/extending.html) in PyTorch to write a wrapper around selective tuning CUDA kernels. It facilitates passing CUDA PyTorch tensors with other arguments to the underlying CUDA kernel implementation with little amount of overhead. The code modular, extendable, easy to develope for future work.

Software requirement is minimal. It only depends on PyTorch and it's dependencies such as ffi and Cython to name a few. It is successfully tested on PyTorch `v0.4.1`.

## Requirements: Hardware

As long as you have a CUDA-enabled GPU card, you should be fine running the code. The overhead of STNet to a typical PyTorch model is only on the GPU card. No CPU or main memory resources will be utilized during the Top-Down pass of STNet.

## Installation

1. Clone the STNet Object Localization repository

```shell
# Clone the repository
git clone https://github.com/mbiparva/stnet-object-localization
```

2. Build STNet custom layers

```shell
# Make the C/CUDA code and build ffi Python wrapper
make all
```

### Prerequisites

* Python 3.6
* PyTorch 0.4.1 (not tested on higher versions)
* CUDA 8.0 or higher

## Preparation

1. Download the validation set (both the xml annotations and jpg images) of the ILSVRC 2012 dataset.
2. Copy them into `($root_dir)/dataset/ILSVRC2012/` under their corresponding directories.
3. If the data is not preprocessed [as described here](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset), then copy the scripts in `($root_dir)/dataset/scripts/` into their corresponding directories. Finally run them each to save each validation files into their category folder. You can specify the dataset path using the system input argument passed to the `stnet_object_localization.py` file.

## Demo: Weakly-Supervised Object Localization

STNet is experimentally evaluated on the weakly-supervised object localization task. In this task, the goal is to achieve object localization in a single-class object categorization regime knowing the ground-truth category labels. This is briefly the procedure STNet goes through:

1. Load and preprocess the input images.
2. Create the network and load the pre-trained model.
2. Bottom-Up feed forward pass is called.
3. Top-Down selective attention pass is called given the ground-truth label.
4. The best bounding box is proposed from the gating activities at the bottom of the network.
5. Network performance is measured:
    1. Label prediction using the prediction scores at the top of the network.
    2. Localization prediction using the gating activities at the bottom of the network.

The localization prediction accuracy is measure using Intersection-over-Union (IoU) metric over 0.5. The label prediction accuracy is measure using Top1 metric.

#### Evaluation Execution Mode

The STNet is implemented to run in the three different evaluation modes. STNet will run in the predicted box evaluation mode by executing the following command:

```shell
python stnet_object_localization.py --exe-mode bbox_eval
```

#### Box Visualization Execution Mode

STNet will run in the predicted box visualization mode by executing the following command. In this mode, the annotation bounding boxes in addition to the predicted one are drawn on the input image.

```shell
python stnet_object_localization.py --exe-mode bbox_viz
```

### Quantitative Evaluation Results

Currently, this implementation is provided for AlexNet architecure provided officially with PyTorch. The AlexNet pre-trained model is loaded from PyTorch model zoo repository. For the default configuration parameters, the metric performance is given in the table below:

|   Model   | Label Accuracy | IoU=0.25 | IoU=0.5 | IoU=0.75 |
|:---------:|:--------------:|:--------:|:-------:|:--------:|
|  AlexNet  |      55.64%    |  77.50%  |  55.30% |   29.10% |
|   VGG-16  |        -       |     -    |    -    |     -    |
| GoogleNet |        -       |     -    |    -    |     -    |

### Qualitative Evaluation Results

We demonstrate the performance of STNet by showing the predicted bounding boxes over the input images. In the figure below, the row from top to bottom represents ground truth, VGG, and GoogleNet boxes. The results are taken from the research paper.

![](http://drive.google.com/uc?export=view&id=1yGx8k1onIWA4XKcoWvaJ8Lyoep_wfPFO)

## Demo: Class Hypothesis Visualization

We further process the gating activities at the bottom layer to with Gaussian blur filter to smooth out the collapse gating tensor and then using heat-map illustration highlight the most activate regions. We call this the class-hypothesis-visualization.

![](http://drive.google.com/uc?export=view&id=1obAKx51V5YmqlOooEhGJcr6u8ZLD0xJA)

#### Class-Hypothesis Visualization Execution Mode

STNet will visualize the class hypothesis derived from the Top-Down pass by executing the following command. In this mode, a heat map of the smoothed gating activities are generated.

```shell
python stnet_object_localization.py --exe-mode ch_viz
```

## Future Work

Currently, this is the preliminary work on the idea of a Top-Down pass in deep neural networks. The questions are:
* Why do we need a Top-Down selection?
* Do we at all need any sort of selection mechanism whether Bottom-Up (Early) or Top-Down (Late)?
* What is the benefit of having ___Early___ selection using layers such as ReLU, Dropout, and Max Pooling over ___Late___ selection?
* What is the best implementation of a Top-Down pass?
* How can we improve on the selection mechanism itself?
* Can we optimize over the Top-Down pass? What would be the criterion then?

These are a number of questions we are faced with in order to fully understand and implement Top-Down process in deep neural networks. STNet is an attempt to show such Top-Down pass can help to select a portion of the learned hierarchical representation using which the network has learned to predict category labels the best.

Like all the other visualization attempts to shed light on the internal representation space of the networks, STNet highlights the relevant regions of category instances that seem most important for the top feature abstraction.

We are working on various aspects of STNet and are going to update the repository with further network architectures.

If you find StNet for Object Localization useful in your research, please consider citing the research paper:

    @InProceedings{Biparva_2017_STNet,
    author = {Biparva, Mahdi and Tsotsos, John},
    title = {STNet: Selective Tuning of Convolutional Networks for Object Localization},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV) Workshops},
    month = {Oct},
    year = {2017}
    }
