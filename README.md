
# Awesome Pruning [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

  

A curated list of neural network pruning and related resources. Inspired by [awesome-deep-vision](https://github.com/kjw0612/awesome-deep-vision), [awesome-adversarial-machine-learning](https://github.com/yenchenlin/awesome-adversarial-machine-learning), [awesome-deep-learning-papers](https://github.com/terryum/awesome-deep-learning-papers) and [Awesome-NAS](https://github.com/D-X-Y/Awesome-NAS).


Please feel free to [pull requests](https://github.com/he-y/awesome-Pruning/pulls) or [open an issue](https://github.com/he-y/awesome-Pruning/issues) to add papers.

  

## Table of Contents

  

- [Neural Network Pruning](#NAS)

- [2019 Venues](#2019)

- [2018 Venues](#2018)

- [2017 Venues](#2017)

- [Previous Venues](#2012-2016)

- [Blogs](#blogs)

- [arXiv](#arxiv)

- [Benchmark on ImageNet](#benchmark-on-imagenet)

  

  

## Neural Architecture Search (NAS)

  

|  Type |  `F` |  `W`  |   |  `Other` |
|:------------|:--------------:|:----------------------:|:----------:|
| Explanation | Filter pruning | Weight pruning | other types |

  

  

### 2019

  

|  Title  | Venue  | Type | Code |
|:--------|:--------:|:--------:|:--------:|
| [Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration](https://arxiv.org/abs/1811.00250) | CVPR (Oral) | |[github](https://github.com/he-y/filter-pruning-geometric-median)|
| [On Implicit Filter Level Sparsity in Convolutional Neural Networks](https://arxiv.org/abs/1811.12495) | - | - | - |
| [Partial Order Pruning: for Best Speed/Accuracy Trade-off in Neural Architecture Search](https://arxiv.org/abs/1903.03777) | CVPR | - | [github](https://github.com/lixincn2015/Partial-Order-Pruning) |


  

### 2018

Add later. 

  

### 2017

Add later.

### 2012-2016




  

### Blogs


  Add later.
  

### arXiv

  Add later.


  

## Benchmark on ImageNet

  

  

| Architecture | Top-1 (%) | Top-5 (%) | Params (M) | +x (M) | GPU | Days |

| ------------------ | --------- | --------- | ---------- | ------ | - | -  |

| [Inception-v1](https://arxiv.org/pdf/1409.4842.pdf) | 30.2  | 10.1  | 6.6  | 1448 | - | -  |

| [MobileNet-v1](https://arxiv.org/abs/1704.04861) | 29.4  | 10.5  | 4.2  | 569  | - | -  |

| [ShuffleNet](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0642.pdf) | 26.3  | - | ~5 | 524  | - | -  |

| [NASNet-A]((http://openaccess.thecvf.com/content_cvpr_2018/papers/Zoph_Learning_Transferable_Architectures_CVPR_2018_paper.pdf)) | 26.0  | 8.4 | 5.3  | 564  | 450 | 3-4  |

| NASNet-B | 27.2  | 8.7 | 5.3  | 488  | 450 | 3-4  |

| NASNet-C | 27.5  | 9.0 | 4.9  | 558  | 450 | 3-4  |

| [AmobebaNet-A](https://arxiv.org/pdf/1802.01548.pdf) | 25.5  | 8.0 | 5.1  | 555  | 450 |  7 |

| AmobebaNet-B | 26.0  | 8.5 | 5.3  | 555  | 450 |  7 |

| AmobebaNet-C | 24.3  | 7.6 | 6.4  | 555  | 450 |  7 |

| [Progressive NAS](http://openaccess.thecvf.com/content_ECCV_2018/papers/Chenxi_Liu_Progressive_Neural_Architecture_ECCV_2018_paper.pdf)  | 25.8  | 8.1 | 5.1  | 588  | 100 | 1.5  |

| [DARTS-V2](https://arxiv.org/abs/1806.09055) | 26.9  | 9.0 | 4.9  | 595  |  1  |  1 |

| [GDAS](http://xuanyidong.com/bibtex/Four-Hours-CVPR19.txt) | 26.0  | 8.5 | 5.3  | 581  |  1  |  0.21 |
