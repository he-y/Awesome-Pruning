# Awesome Pruning [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of neural network pruning and related resources. Inspired by [awesome-deep-vision](https://github.com/kjw0612/awesome-deep-vision), [awesome-adversarial-machine-learning](https://github.com/yenchenlin/awesome-adversarial-machine-learning), [awesome-deep-learning-papers](https://github.com/terryum/awesome-deep-learning-papers) and [Awesome-NAS](https://github.com/D-X-Y/Awesome-NAS).

Please feel free to [pull requests](https://github.com/he-y/awesome-Pruning/pulls) or [open an issue](https://github.com/he-y/awesome-Pruning/issues) to add papers.

## Table of Contents

- [Type of Pruning](#type-of-pruning)

- [2020 Venues](#2020)

- [2019 Venues](#2019)

- [2018 Venues](#2018)

- [2017 Venues](#2017)

- [2016 Venues](#2016)

- [2015 Venues](#2015)

### Type of Pruning

| Type        | `F`            | `W`            | `Other`     |
|:----------- |:--------------:|:--------------:|:-----------:|
| Explanation | Filter pruning | Weight pruning | other types |

### 2020

| Title                                                                                                                            | Venue | Type    | Code |
|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|
| [HRank: Filter Pruning using High-Rank Feature Map](https://arxiv.org/abs/2002.10179) | CVPR | `F`     | [Pytorch(Author)](https://github.com/lmbxmu/HRank)   |
| [Neural Network Pruning with Residual-Connections and Limited-Data](https://arxiv.org/abs/1911.08114) | CVPR | `F`     | -  |
| [DMCP: Differentiable Markov Channel Pruning for Neural Networks](https://arxiv.org/abs/2005.03354) | CVPR | `F`     | [TensorFlow(Author)](https://github.com/zx55/dmcp)   |
| [Group Sparsity: The Hinge Between Filter Pruning and Decomposition for Network Compression](https://arxiv.org/abs/2003.08935) | CVPR | `F`     | [PyTorch(Author)](https://github.com/ofsoundof/group_sparsity)   |
| [Comparing Rewinding and Fine-tuning in Neural Network Pruning](https://arxiv.org/abs/2003.02389) | ICLR **(Oral)** | `WF`     | [TensorFlow(Author)](https://github.com/lottery-ticket/code)   |
| [A Signal Propagation Perspective for Pruning Neural Networks at Initialization](https://arxiv.org/abs/1906.06307) | ICLR **(Spotlight)** | `W`     | -  |
| [One-Shot Pruning of Recurrent Neural Networks by Jacobian Spectrum Evaluation](https://arxiv.org/abs/1912.00120) | ICLR   | `W`     | -  |
| [Lookahead: A Far-sighted Alternative of Magnitude-based Pruning](https://arxiv.org/abs/2002.04809) | ICLR   | `W`     | [PyTorch(Author)](https://github.com/alinlab/lookahead_pruning)  |
| [Dynamic Model Pruning with Feedback](https://openreview.net/forum?id=SJem8lSFwB) | ICLR   | `WF`     | -  |
| [Provable Filter Pruning for Efficient Neural Networks](https://arxiv.org/abs/1911.07412) | ICLR   | `F`     | -  |
| [Data-Independent Neural Pruning via Coresets](https://arxiv.org/abs/1907.04018) | ICLR  | `W`     | -    |
| [AutoCompress: An Automatic DNN Structured Pruning Framework for Ultra-High Compression Rates](https://arxiv.org/abs/1907.03141) | AAAI  | `F`     | -    |
| [DARB: A Density-Aware Regular-Block Pruning for Deep Neural Networks](http://arxiv.org/abs/1911.08020)                          | AAAI  | `Other` | -    |
| [Pruning from Scratch](http://arxiv.org/abs/1909.12579)                                                                          | AAAI  | `Other` | -    |

### 2019

| Title    | Venue       | Type    | Code     |
|:-------|:--------:|:-------:|:-------:|
| [Network Pruning via Transformable Architecture Search](https://arxiv.org/abs/1905.09717)                                                                                                                        | NeurIPS         | `F`     | [PyTorch(Author)](https://github.com/D-X-Y/NAS-Projects)                              |
| [Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks](https://arxiv.org/abs/1909.08174)                                                                             | NeurIPS         | `F`     | [PyTorch(Author)](https://github.com/youzhonghui/gate-decorator-pruning)              |
| [Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask](https://arxiv.org/abs/1905.01067)                                                                                                              | NeurIPS         | `W`     | [TensorFlow(Author)](https://github.com/uber-research/deconstructing-lottery-tickets) |
| [One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers](https://arxiv.org/abs/1906.02773)                                                                       | NeurIPS         | `W`     | -                                                                                     |
| [Global Sparse Momentum SGD for Pruning Very Deep Neural Networks](https://arxiv.org/abs/1909.12778)                                                                                                             | NeurIPS         | `W`     | [PyTorch(Author)](https://github.com/DingXiaoH/GSM-SGD)                               |
| [AutoPrune: Automatic Network Pruning by Regularizing Auxiliary Parameters](https://papers.nips.cc/paper/9521-autoprune-automatic-network-pruning-by-regularizing-auxiliary-parameters)                          | NeurIPS         | `W`     | -                                                                                     |
| [Model Compression with Adversarial Robustness: A Unified Optimization Framework](https://arxiv.org/abs/1902.03538)                                                                                              | NeurIPS         | `Other` | [PyTorch(Author)](https://github.com/TAMU-VITA/ATMC)                                  |
| [MetaPruning: Meta Learning for Automatic Neural Network Channel Pruning](https://arxiv.org/abs/1903.10258)                                                                                                      | ICCV            | `F`     | [PyTorch(Author)](https://github.com/liuzechun/MetaPruning)                           |
| [Accelerate CNN via Recursive Bayesian Pruning](https://arxiv.org/abs/1812.00353)                                                                                                                                | ICCV            | `F`     | -                        |
| [Adversarial Robustness vs Model Compression, or Both?](https://arxiv.org/abs/1903.12561)                                                                                                                        | ICCV            | `W`     | [PyTorch(Author)](https://github.com/yeshaokai/Robustness-Aware-Pruning-ADMM)         |
| [Learning Filter Basis for Convolutional Neural Network Compression](https://arxiv.org/abs/1908.08932)                                                                                                           | ICCV            | `Other` | -                                                                                     |
| [Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration](https://arxiv.org/abs/1811.00250)                                                                                      | CVPR **(Oral)** | `F`     | [PyTorch(Author)](https://github.com/he-y/filter-pruning-geometric-median)            |
| [Towards Optimal Structured CNN Pruning via Generative Adversarial Learning](https://arxiv.org/abs/1903.09291)                                                                                                   | CVPR            | `F`     | [PyTorch(Author)](https://github.com/ShaohuiLin/GAL)                                  |
| [Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure](https://arxiv.org/abs/1904.03837)                                                                                      | CVPR            | `F`     | [PyTorch(Author)](https://github.com/ShawnDing1994/Centripetal-SGD)                   |
| [On Implicit Filter Level Sparsity in Convolutional Neural Networks](https://arxiv.org/abs/1811.12495), [Extension1](https://arxiv.org/abs/1905.04967), [Extension2](https://openreview.net/forum?id=rylVvNS3hE) | CVPR            | `F`     | [PyTorch(Author)](https://github.com/mehtadushy/SelecSLS-Pytorch)                     |
| [Structured Pruning of Neural Networks with Budget-Aware Regularization](https://arxiv.org/abs/1811.09332)                                                                                                       | CVPR            | `F`     | -                                                                                     |
| [Importance Estimation for Neural Network Pruning](http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf)                                                                                             | CVPR            | `F`     | [PyTorch(Author)](https://github.com/NVlabs/Taylor_pruning)                           |
| [OICSR: Out-In-Channel Sparsity Regularization for Compact Deep Neural Networks](https://arxiv.org/abs/1905.11664)                                                                                               | CVPR            | `F`     | -                                                                                     |
| [Partial Order Pruning: for Best Speed/Accuracy Trade-off in Neural Architecture Search](https://arxiv.org/abs/1903.03777)                                                                                       | CVPR            | `Other` | [TensorFlow(Author)](https://github.com/lixincn2015/Partial-Order-Pruning)            |
| [Variational Convolutional Neural Network Pruning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_Variational_Convolutional_Neural_Network_Pruning_CVPR_2019_paper.pdf)                              | CVPR            | -       | -                                                                                     |
| [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)                                                                                                     | ICLR **(Best)** | `W`     | [TensorFlow(Author)](https://github.com/google-research/lottery-ticket-hypothesis)                |
| [Rethinking the Value of Network Pruning](https://arxiv.org/abs/1810.05270)                                                                                                                                      | ICLR            | `F`     | [PyTorch(Author)](https://github.com/Eric-mingjie/rethinking-network-pruning)         |
| [Dynamic Channel Pruning: Feature Boosting and Suppression](https://arxiv.org/abs/1810.05331)                                                                                                                    | ICLR            | `F`     | [TensorFlow(Author)](https://github.com/deep-fry/mayo)                                |
| [SNIP: Single-shot Network Pruning based on Connection Sensitivity](https://arxiv.org/abs/1810.02340)                                                                                                            | ICLR            | `F`     | [TensorFLow(Author)](https://github.com/namhoonlee/snip-public)                       |
| [Dynamic Sparse Graph for Efficient Deep Learning](https://arxiv.org/abs/1810.00859)                                                                                                                             | ICLR            | `F`     | [CUDA(3rd)](https://github.com/mtcrawshaw/dynamic-sparse-graph)                       |
| [Collaborative Channel Pruning for Deep Networks](http://proceedings.mlr.press/v97/peng19c.html)                                                                                                                 | ICML            | `F`     | -                                                                                     |
| [Approximated Oracle Filter Pruning for Destructive CNN Width Optimization github](https://arxiv.org/abs/1905.04748)                                                                                             | ICML            | `F`     | -                                                                                     |
| [EigenDamage: Structured Pruning in the Kronecker-Factored Eigenbasis4](https://arxiv.org/abs/1905.05934)                                                                                                        | ICML            | `W`     | [PyTorch(Author)](https://github.com/alecwangcq/EigenDamage-Pytorch)                  |

### 2018

| Title    | Venue       | Type    | Code     |
|:-------|:--------:|:-------:|:-------:|
| [Rethinking the Smaller-Norm-Less-Informative Assumption in Channel Pruning of Convolution Layers](https://arxiv.org/abs/1802.00124)                                              | ICLR    | `F`  | [TensorFlow(Author)](https://github.com/bobye/batchnorm_prune), [PyTorch(3rd)](https://github.com/jack-willturner/batchnorm-pruning) |
| [To prune, or not to prune: exploring the efficacy of pruning for model compression](https://arxiv.org/abs/1710.01878)                                                            | ICLR    | `W`  | -                                                                                                                                    |
| [Discrimination-aware Channel Pruning for Deep Neural Networks](https://arxiv.org/abs/1810.11809)                                                                                 | NeurIPS | `F`  | [TensorFlow(Author)](https://github.com/SCUT-AILab/DCP)                                                                              |
| [Frequency-Domain Dynamic Pruning for Convolutional Neural Networks](https://papers.NeurIPS.cc/paper/7382-frequency-domain-dynamic-pruning-for-convolutional-neural-networks.pdf) | NeurIPS | `W`  | -                                                                                                                                    |
| [Amc: Automl for model compression and acceleration on mobile devices](https://arxiv.org/abs/1802.03494)                                                                          | ECCV    | `F`  | [TensorFlow(3rd)](https://github.com/Tencent/PocketFlow#channel-pruning)                                                             |
| [Data-Driven Sparse Structure Selection for Deep Neural Networks](https://arxiv.org/abs/1707.01213)                                                                               | ECCV    | `F`  | [MXNet(Author)](https://github.com/TuSimple/sparse-structure-selection)                                                              |
| [Coreset-Based Neural Network Compression](https://arxiv.org/abs/1807.09810)                                                                                                      | ECCV    | `F`  | [PyTorch(Author)](https://github.com/metro-smiles/CNN_Compression)                                                                   |
| [Constraint-Aware Deep Neural Network Compression](http://www.sfu.ca/~ftung/papers/constraintaware_eccv18.pdf)                                                                    | ECCV    | `W`  | [SkimCaffe(Author)](https://github.com/ChanganVR/ConstraintAwareCompression)                                                         |
| [A Systematic DNN Weight Pruning Framework using Alternating Direction Method of Multipliers](https://arxiv.org/abs/1804.03294)                                                   | ECCV    | `W`  | [Caffe(Author)](https://github.com/KaiqiZhang/admm-pruning)                                                                          |
| [PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning](https://arxiv.org/abs/1711.05769)                                                                       | CVPR    | `F`  | [PyTorch(Author)](https://github.com/arunmallya/packnet)                                                                             |
| [NISP: Pruning Networks using Neuron Importance Score Propagation](https://arxiv.org/abs/1711.05908)                                                                              | CVPR    | `F`  | -                                                                                                                                    |
| [CLIP-Q: Deep Network Compression Learning by In-Parallel Pruning-Quantization](http://www.sfu.ca/~ftung/papers/clipq_cvpr18.pdf)                                                 | CVPR    | `W`  | -                                                                                                                                    |
| [“Learning-Compression” Algorithms for Neural Net Pruning](http://faculty.ucmerced.edu/mcarreira-perpinan/papers/cvpr18.pdf)                                                      | CVPR    | `W`  | -                                                                                                                                    |
| [Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks](https://arxiv.org/abs/1808.06866)                                                                       | IJCAI   | `F`  | [PyTorch(Author)](https://github.com/he-y/soft-filter-pruning)                                                                       |
| [Accelerating Convolutional Networks via Global & Dynamic Filter Pruning](https://www.ijcai.org/proceedings/2018/0336.pdf)                                                        | IJCAI   | `F`  | -                                                                                                                                    |

### 2017

| Title    | Venue       | Type    | Code     |
|:-------|:--------:|:-------:|:-------:|
| [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)                                              | ICLR    | `F`  | [PyTorch(3rd)](https://github.com/Eric-mingjie/rethinking-network-pruning/tree/master/imagenet/l1-norm-pruning)       |
| [Pruning Convolutional Neural Networks for Resource Efficient Inference](https://arxiv.org/abs/1611.06440)              | ICLR    | `F`  | [TensorFlow(3rd)](https://github.com/Tencent/PocketFlow#channel-pruning)                                              |
| [Net-Trim: Convex Pruning of Deep Neural Networks with Performance Guarantee](https://arxiv.org/abs/1611.05162)         | NeurIPS | `W`  | [TensorFlow(Author)](https://github.com/DNNToolBox/Net-Trim-v1)                                                       |
| [Learning to Prune Deep Neural Networks via Layer-wise Optimal Brain Surgeon](https://arxiv.org/abs/1705.07565)         | NeurIPS | `W`  | [PyTorch(Author)](https://github.com/csyhhu/L-OBS)                                                                    |
| [Runtime Neural Pruning](https://papers.NeurIPS.cc/paper/6813-runtime-neural-pruning)                                   | NeurIPS | `F`  | -                                                                                                                     |
| [Designing Energy-Efficient Convolutional Neural Networks using Energy-Aware Pruning](https://arxiv.org/abs/1611.05128) | CVPR    | `F`  | -                                                                                                                     |
| [ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression](https://arxiv.org/abs/1707.06342)           | ICCV    | `F`  | [Caffe(Author)](https://github.com/Roll920/ThiNet), [PyTorch(3rd)](https://github.com/tranorrepository/reprod-thinet) |
| [Channel pruning for accelerating very deep neural networks](https://arxiv.org/abs/1707.06168)                          | ICCV    | `F`  | [Caffe(Author)](https://github.com/yihui-he/channel-pruning)                                                          |
| [Learning Efficient Convolutional Networks Through Network Slimming](https://arxiv.org/abs/1708.06519)                  | ICCV    | `F`  | [PyTorch(Author)](https://github.com/Eric-mingjie/network-slimming)                                                   |

### 2016

| Title    | Venue       | Type    | Code     |
|:-------|:--------:|:-------:|:-------:|
| [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149) | ICLR **(Best)** | `W`  | [Caffe(Author)](https://github.com/songhan/Deep-Compression-AlexNet) |
| [Dynamic Network Surgery for Efficient DNNs](https://arxiv.org/abs/1608.04493)                                                               | NeurIPS         | `W`  | [Caffe(Author)](https://github.com/yiwenguo/Dynamic-Network-Surgery) |

### 2015

| Title    | Venue       | Type    | Code     |
|:-------|:--------:|:-------:|:-------:|
| [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626) | NeurIPS | `W`  | [PyTorch(3rd)](https://github.com/jack-willturner/DeepCompression-PyTorch) |

## Related Repo

[Awesome-model-compression-and-acceleration](https://github.com/memoiry/Awesome-model-compression-and-acceleration)

[EfficientDNNs](https://github.com/MingSun-Tse/EfficientDNNs)

[Embedded-Neural-Network](https://github.com/ZhishengWang/Embedded-Neural-Network)

[awesome-AutoML-and-Lightweight-Models](https://github.com/guan-yuan/awesome-AutoML-and-Lightweight-Models)

[Model-Compression-Papers](https://github.com/chester256/Model-Compression-Papers)

[knowledge-distillation-papers](https://github.com/lhyfst/knowledge-distillation-papers)

[Network-Speed-and-Compression](https://github.com/mrgloom/Network-Speed-and-Compression)
