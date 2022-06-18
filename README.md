# Awesome Pruning [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of neural network pruning and related resources. Inspired by [awesome-deep-vision](https://github.com/kjw0612/awesome-deep-vision), [awesome-adversarial-machine-learning](https://github.com/yenchenlin/awesome-adversarial-machine-learning), [awesome-deep-learning-papers](https://github.com/terryum/awesome-deep-learning-papers) and [Awesome-NAS](https://github.com/D-X-Y/Awesome-NAS).

Please feel free to [pull requests](https://github.com/he-y/awesome-Pruning/pulls) or [open an issue](https://github.com/he-y/awesome-Pruning/issues) to add papers.

## Table of Contents

- [Type of Pruning](#type-of-pruning)
- [2021 Venues](#2021)

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

### 2021

| Title                                                                                                                            | Venue | Type    | Code |
|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|
| [A Probabilistic Approach to Neural Network Pruning](https://arxiv.org/abs/2105.10065) | ICML | `F`     | -   |
| [Accelerate CNNs from Three Dimensions: A Comprehensive Pruning Framework](https://arxiv.org/abs/2010.04879) | ICML | `F`     | -  |
| [Group Fisher Pruning for Practical Network Compression](https://arxiv.org/abs/2108.00708) | ICML | `F`     | [PyTorch(Author)](https://github.com/jshilong/FisherPruning)   |
| [On the Predictability of Pruning Across Scales](https://arxiv.org/abs/2006.10621) | ICML | `W`     | -   |
| [Towards Compact CNNs via Collaborative Compression](https://arxiv.org/abs/2105.11228) | CVPR | `F`     | [PyTorch(Author)](https://github.com/liuguoyou/Towards-Compact-CNNs-via-Collaborative-Compression)   |
| [Content-Aware GAN Compression](https://arxiv.org/abs/2104.02244) | CVPR | `F`     | [PyTorch(Author)](https://github.com/lychenyoko/content-aware-gan-compression)   |
| [Permute, Quantize, and Fine-tune: Efficient Compression of Neural Networks](https://arxiv.org/abs/2010.15703) | CVPR | `F`     | [PyTorch(Author)](https://github.com/uber-research/permute-quantize-finetune)   |
| [NPAS: A Compiler-aware Framework of Unified Network Pruning andArchitecture Search for Beyond Real-Time Mobile Acceleration](https://arxiv.org/abs/2012.00596) | CVPR | `F`     | -  |
| [Network Pruning via Performance Maximization](https://openaccess.thecvf.com/content/CVPR2021/papers/Gao_Network_Pruning_via_Performance_Maximization_CVPR_2021_paper.pdf) | CVPR | `F`     | -  |
| [Convolutional Neural Network Pruning with Structural Redundancy Reduction](https://arxiv.org/abs/2104.03438) | CVPR | `F`     |-|
| [Manifold Regularized Dynamic Network Pruning](https://arxiv.org/abs/2103.05861) | CVPR | `F`     | -  |
| [Joint-DetNAS: Upgrade Your Detector with NAS, Pruning and Dynamic Distillation](https://arxiv.org/abs/2105.12971) | CVPR | `FO`     | -  |
| [A Gradient Flow Framework For Analyzing Network Pruning](https://openreview.net/forum?id=rumv7QmLUue) | ICLR | `F`     | [PyTorch(Author)](https://github.com/EkdeepSLubana/flowandprune)   |
| [Neural Pruning via Growing Regularization](https://openreview.net/forum?id=o966_Is_nPA) | ICLR | `F`     | [PyTorch(Author)](https://github.com/MingSun-Tse/Regularization-Pruning)   |
| [ChipNet: Budget-Aware Pruning with Heaviside Continuous Approximations](https://openreview.net/forum?id=xCxXwTzx4L1) | ICLR | `F`     | [PyTorch(Author)](https://github.com/transmuteAI/ChipNet)   |
| [Network Pruning That Matters: A Case Study on Retraining Variants](https://openreview.net/forum?id=Cb54AMqHQFP) | ICLR | `F`     | [PyTorch(Author)](https://github.com/lehduong/NPTM)   |
| [Multi-Prize Lottery Ticket Hypothesis: Finding Accurate Binary Neural Networks by Pruning A Randomly Weighted Network](https://openreview.net/forum?id=U_mat0b9iv) | ICLR | `W`     | [PyTorch(Author)](https://github.com/chrundle/biprop)   |
| [Layer-adaptive Sparsity for the Magnitude-based Pruning](https://openreview.net/forum?id=H6ATjJ0TKdf) | ICLR | `W`     | [PyTorch(Author)](https://github.com/jaeho-lee/layer-adaptive-sparsity)   |
| [Pruning Neural Networks at Initialization: Why Are We Missing the Mark?](https://openreview.net/forum?id=Ig-VyQc-MLK) | ICLR | `W`     | - |
| [Robust Pruning at Initialization](https://openreview.net/forum?id=vXj_ucZQ4hA) | ICLR | `W`     |  -  |

### 2020

| Title                                                                                                                            | Venue | Type    | Code |
|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|
| [HYDRA: Pruning Adversarially Robust Neural Networks](https://arxiv.org/abs/2002.10509) | NeurIPS | `W`     | [PyTorch(Author)](https://github.com/inspire-group/hydra)   |
| [Logarithmic Pruning is All You Need](https://arxiv.org/abs/2006.12156) | NeurIPS | `W`     | -   |
| [Directional Pruning of Deep Neural Networks](https://arxiv.org/abs/2006.09358) | NeurIPS | `W`     |  - |
| [Movement Pruning: Adaptive Sparsity by Fine-Tuning](https://arxiv.org/abs/2005.07683) | NeurIPS | `W`     | [PyTorch(Author)](https://github.com/huggingface/block_movement_pruning) |
| [Sanity-Checking Pruning Methods: Random Tickets can Win the Jackpot](https://arxiv.org/abs/2009.11094) | NeurIPS | `W`     | [PyTorch(Author)](https://github.com/JingtongSu/sanity-checking-pruning) |
| [Neuron Merging: Compensating for Pruned Neurons](https://arxiv.org/abs/2010.13160) | NeurIPS | `F`     | [PyTorch(Author)](https://github.com/friendshipkim/neuron-merging)  |
| [Neuron-level Structured Pruning using Polarization Regularizer](https://papers.nips.cc/paper/2020/file/703957b6dd9e3a7980e040bee50ded65-Paper.pdf) | NeurIPS | `F`     |  [PyTorch(Author)](https://github.com/polarizationpruning/PolarizationPruning)   |
| [SCOP: Scientific Control for Reliable Neural Network Pruning](https://arxiv.org/abs/2010.10732) | NeurIPS | `F`     |  [PyTorch(Author)](https://github.com/yehuitang/Pruning/tree/master/SCOP_NeurIPS2020)  |
| [Storage Efficient and Dynamic Flexible Runtime Channel Pruning via Deep Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/hash/a914ecef9c12ffdb9bede64bb703d877-Abstract.html) | NeurIPS | `F`     |  - |
| [The Generalization-Stability Tradeoff In Neural Network Pruning](https://arxiv.org/abs/1906.03728) | NeurIPS | `F`     | [PyTorch(Author)](https://github.com/bbartoldson/GeneralizationStabilityTradeoff) |
| [Pruning Filter in Filter](https://arxiv.org/abs/2009.14410) | NeurIPS | `Other`     | [PyTorch(Author)](https://github.com/fxmeng/Pruning-Filter-in-Filter)   |
| [Position-based Scaled Gradient for Model Quantization and Pruning](https://arxiv.org/abs/2005.11035) | NeurIPS | `Other`     | [PyTorch(Author)](https://github.com/Jangho-Kim/PSG-pytorch) |
| [Bayesian Bits: Unifying Quantization and Pruning](https://arxiv.org/abs/2005.07093) | NeurIPS | `Other`     | -  |
| [Pruning neural networks without any data by iteratively conserving synaptic flow](https://arxiv.org/abs/2006.05467) | NeurIPS | `Other`     |  [PyTorch(Author)](https://github.com/ganguli-lab/Synaptic-Flow)   |
| [EagleEye: Fast Sub-net Evaluation for Efficient Neural Network Pruning](https://arxiv.org/abs/2007.02491) | ECCV **(Oral)** | `F`     | [PyTorch(Author)](https://github.com/anonymous47823493/EagleEye)   |
| [DSA: More Efficient Budgeted Pruning via Differentiable Sparsity Allocation](https://arxiv.org/abs/2004.02164) | ECCV  | `F`     |- |
| [DHP: Differentiable Meta Pruning via HyperNetworks](https://arxiv.org/abs/2003.13683) | ECCV  | `F`     |[PyTorch(Author)](https://github.com/ofsoundof/dhp)  |
| [Meta-Learning with Network Pruning](https://arxiv.org/abs/2007.03219) | ECCV  | `W`     | - |
| [Accelerating CNN Training by Pruning Activation Gradients](https://arxiv.org/abs/1908.00173) | ECCV  | `W`     | - |
| [DA-NAS: Data Adapted Pruning for Efficient Neural Architecture Search](https://arxiv.org/abs/2003.12563) | ECCV  | `Other`     | - |
| [Differentiable Joint Pruning and Quantization for Hardware Efficiency](https://arxiv.org/abs/2007.10463) | ECCV  | `Other`     | - |
| [Channel Pruning via Automatic Structure Search](https://arxiv.org/abs/2001.08565) | IJCAI | `F`     | [PyTorch(Author)](https://github.com/lmbxmu/ABCPruner)   |
| [Adversarial Neural Pruning with Latent Vulnerability Suppression](https://arxiv.org/abs/1908.04355) | ICML | `W`     | -   |
| [Proving the Lottery Ticket Hypothesis: Pruning is All You Need](https://arxiv.org/abs/2002.00585) | ICML | `W`     | -   |
| [Soft Threshold Weight Reparameterization for Learnable Sparsity](https://arxiv.org/abs/2002.03231) | ICML | `WF`     | [Pytorch(Author)](https://github.com/RAIVNLab/STR)   |
| [Network Pruning by Greedy Subnetwork Selection](https://arxiv.org/abs/2003.01794) | ICML | `F`     | -   |
| [Operation-Aware Soft Channel Pruning using Differentiable Masks](https://arxiv.org/abs/2007.03938) | ICML | `F`     | -   |
| [DropNet: Reducing Neural Network Complexity via Iterative Pruning](https://proceedings.mlr.press/v119/tan20a.html) | ICML | `F`     | -   |
| [Towards Efficient Model Compression via Learned Global Ranking](https://arxiv.org/abs/1904.12368) | CVPR  **(Oral)**| `F`     | [Pytorch(Author)](https://github.com/cmu-enyac/LeGR)   |
| [HRank: Filter Pruning using High-Rank Feature Map](https://arxiv.org/abs/2002.10179) | CVPR **(Oral)** | `F`     | [Pytorch(Author)](https://github.com/lmbxmu/HRank)   |
| [Neural Network Pruning with Residual-Connections and Limited-Data](https://arxiv.org/abs/1911.08114) | CVPR **(Oral)** | `F`     | -  |
| [Multi-Dimensional Pruning: A Unified Framework for Model Compression](http://openaccess.thecvf.com/content_CVPR_2020/html/Guo_Multi-Dimensional_Pruning_A_Unified_Framework_for_Model_Compression_CVPR_2020_paper.html) | CVPR **(Oral)** | `WF`     | -  |
| [DMCP: Differentiable Markov Channel Pruning for Neural Networks](https://arxiv.org/abs/2005.03354) | CVPR **(Oral)** | `F`     | [TensorFlow(Author)](https://github.com/zx55/dmcp)   |
| [Group Sparsity: The Hinge Between Filter Pruning and Decomposition for Network Compression](https://arxiv.org/abs/2003.08935) | CVPR | `F`     | [PyTorch(Author)](https://github.com/ofsoundof/group_sparsity)   |
| [Few Sample Knowledge Distillation for Efficient Network Compression](https://arxiv.org/abs/1812.01839) | CVPR   | `F`     | -  |
| [Discrete Model Compression With Resource Constraint for Deep Neural Networks](http://openaccess.thecvf.com/content_CVPR_2020/html/Gao_Discrete_Model_Compression_With_Resource_Constraint_for_Deep_Neural_Networks_CVPR_2020_paper.html) | CVPR   | `F`     | -  |
| [Structured Compression by Weight Encryption for Unstructured Pruning and Quantization](https://arxiv.org/abs/1905.10138) | CVPR   | `W`     | -  |
| [Learning Filter Pruning Criteria for Deep Convolutional Neural Networks Acceleration](http://openaccess.thecvf.com/content_CVPR_2020/html/He_Learning_Filter_Pruning_Criteria_for_Deep_Convolutional_Neural_Networks_Acceleration_CVPR_2020_paper.html) | CVPR   | `F`     | -  |
| [APQ: Joint Search for Network Architecture, Pruning and Quantization Policy](https://arxiv.org/abs/2006.08509) | CVPR   | `F`     | -  |
| [Comparing Rewinding and Fine-tuning in Neural Network Pruning](https://arxiv.org/abs/2003.02389) | ICLR **(Oral)** | `WF`     | [TensorFlow(Author)](https://github.com/lottery-ticket/rewinding-iclr20-public)   |
| [A Signal Propagation Perspective for Pruning Neural Networks at Initialization](https://arxiv.org/abs/1906.06307) | ICLR **(Spotlight)** | `W`     | -  |
| [ProxSGD: Training Structured Neural Networks under Regularization and Constraints](https://openreview.net/forum?id=HygpthEtvr) | ICLR   | `W`     | [TF+PT(Author)](https://github.com/optyang/proxsgd)  |
| [One-Shot Pruning of Recurrent Neural Networks by Jacobian Spectrum Evaluation](https://arxiv.org/abs/1912.00120) | ICLR   | `W`     | -  |
| [Lookahead: A Far-sighted Alternative of Magnitude-based Pruning](https://arxiv.org/abs/2002.04809) | ICLR   | `W`     | [PyTorch(Author)](https://github.com/alinlab/lookahead_pruning)  |
| [Dynamic Model Pruning with Feedback](https://openreview.net/forum?id=SJem8lSFwB) | ICLR   | `WF`     | -  |
| [Provable Filter Pruning for Efficient Neural Networks](https://arxiv.org/abs/1911.07412) | ICLR   | `F`     | -  |
| [Data-Independent Neural Pruning via Coresets](https://arxiv.org/abs/1907.04018) | ICLR  | `W`     | -    |
| [AutoCompress: An Automatic DNN Structured Pruning Framework for Ultra-High Compression Rates](https://arxiv.org/abs/1907.03141) | AAAI  | `F`     | -    |
| [DARB: A Density-Aware Regular-Block Pruning for Deep Neural Networks](http://arxiv.org/abs/1911.08020)                          | AAAI  | `Other` | -    |
| [Pruning from Scratch](http://arxiv.org/abs/1909.12579)                                                                          | AAAI  | `Other` | -    |
| [Reborn filters: Pruning convolutional neural networks with limited data](https://ojs.aaai.org/index.php/AAAI/article/view/6058)                                                                          | AAAI  | `F` | -    |

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
| [SNIP: Single-shot Network Pruning based on Connection Sensitivity](https://arxiv.org/abs/1810.02340)                                                                                                            | ICLR            | `W`     | [TensorFLow(Author)](https://github.com/namhoonlee/snip-public)                       |
| [Dynamic Sparse Graph for Efficient Deep Learning](https://arxiv.org/abs/1810.00859)                                                                                                                             | ICLR            | `F`     | [CUDA(3rd)](https://github.com/mtcrawshaw/dynamic-sparse-graph)                       |
| [Collaborative Channel Pruning for Deep Networks](http://proceedings.mlr.press/v97/peng19c.html)                                                                                                                 | ICML            | `F`     | -                                                                                     |
| [Approximated Oracle Filter Pruning for Destructive CNN Width Optimization github](https://arxiv.org/abs/1905.04748)                                                                                             | ICML            | `F`     | -                                                                                     |
| [EigenDamage: Structured Pruning in the Kronecker-Factored Eigenbasis4](https://arxiv.org/abs/1905.05934)                                                                                                        | ICML            | `W`     | [PyTorch(Author)](https://github.com/alecwangcq/EigenDamage-Pytorch)                  |
| [COP: Customized Deep Model Compression via Regularized Correlation-Based Filter-Level Pruning](https://arxiv.org/abs/1906.10337)                                                                                                        | IJCAI            | `F`     | [Tensorflow(Author)](https://github.com/ZJULearning/COP)                  |

### 2018

| Title    | Venue       | Type    | Code     |
|:-------|:--------:|:-------:|:-------:|
| [Rethinking the Smaller-Norm-Less-Informative Assumption in Channel Pruning of Convolution Layers](https://arxiv.org/abs/1802.00124)                                              | ICLR    | `F`  | [TensorFlow(Author)](https://github.com/bobye/batchnorm_prune), [PyTorch(3rd)](https://github.com/jack-willturner/batchnorm-pruning) |
| [To prune, or not to prune: exploring the efficacy of pruning for model compression](https://arxiv.org/abs/1710.01878)                                                            | ICLR    | `W`  | -                                                                                                                                    |
| [Discrimination-aware Channel Pruning for Deep Neural Networks](https://arxiv.org/abs/1810.11809)                                                                                 | NeurIPS | `F`  | [TensorFlow(Author)](https://github.com/SCUT-AILab/DCP)                                                                              |
| [Frequency-Domain Dynamic Pruning for Convolutional Neural Networks](https://papers.NeurIPS.cc/paper/7382-frequency-domain-dynamic-pruning-for-convolutional-neural-networks.pdf) | NeurIPS | `W`  | -                                                                                                                                    |
| [Learning Sparse Neural Networks via Sensitivity-Driven Regularization](https://arxiv.org/pdf/1810.11764.pdf)                                                                     | NeurIPS | `WF` | -                                                                                                                                    |
| [Amc: Automl for model compression and acceleration on mobile devices](https://arxiv.org/abs/1802.03494)                                                                          | ECCV    | `F`  | [TensorFlow(3rd)](https://github.com/Tencent/PocketFlow#channel-pruning)                                                             |
| [Data-Driven Sparse Structure Selection for Deep Neural Networks](https://arxiv.org/abs/1707.01213)                                                                               | ECCV    | `F`  | [MXNet(Author)](https://github.com/TuSimple/sparse-structure-selection)                                                              |
| [Coreset-Based Neural Network Compression](https://arxiv.org/abs/1807.09810)                                                                                                      | ECCV    | `F`  | [PyTorch(Author)](https://github.com/metro-smiles/CNN_Compression)                                                                   |
| [Constraint-Aware Deep Neural Network Compression](https://openaccess.thecvf.com/content_ECCV_2018/html/Changan_Chen_Constraints_Matter_in_ECCV_2018_paper.html)                                                                    | ECCV    | `W`  | [SkimCaffe(Author)](https://github.com/ChanganVR/ConstraintAwareCompression)                                                         |
| [A Systematic DNN Weight Pruning Framework using Alternating Direction Method of Multipliers](https://arxiv.org/abs/1804.03294)                                                   | ECCV    | `W`  | [Caffe(Author)](https://github.com/KaiqiZhang/admm-pruning)                                                                          |
| [PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning](https://arxiv.org/abs/1711.05769)                                                                       | CVPR    | `F`  | [PyTorch(Author)](https://github.com/arunmallya/packnet)                                                                             |
| [NISP: Pruning Networks using Neuron Importance Score Propagation](https://arxiv.org/abs/1711.05908)                                                                              | CVPR    | `F`  | -                                                                                                                                    |
| [CLIP-Q: Deep Network Compression Learning by In-Parallel Pruning-Quantization](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tung_CLIP-Q_Deep_Network_CVPR_2018_paper.pdf)                                                 | CVPR    | `W`  | -                                                                                                                                    |
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
