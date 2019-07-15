# Awesome NAS [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of neural architecture search and related resources. Inspired by [awesome-deep-vision](https://github.com/kjw0612/awesome-deep-vision), [awesome-adversarial-machine-learning](https://github.com/yenchenlin/awesome-adversarial-machine-learning), [awesome-deep-learning-papers](https://github.com/terryum/awesome-deep-learning-papers), and [awesome-architecture-search](https://github.com/markdtw/awesome-architecture-search).

Please feel free to [pull requests](https://github.com/D-X-Y/awesome-NAS/pulls) or [open an issue](https://github.com/D-X-Y/awesome-NAS/issues) to add papers.

## Table of Contents

- [Neural Architecture Search](#NAS)
  - [2019 Venues](#2019)
  - [2018 Venues](#2018)
  - [2017 Venues](#2017)
  - [Previous Venues](#2012-2016)
  - [Blogs](#blogs)
  - [arXiv](#arxiv)
- [Benchmark on ImageNet](#benchmark-on-imagenet)


## Neural Architecture Search (NAS)

|      Type   |        G       |                  RL    |            EA           |        PD              |    Other   |
|:------------|:--------------:|:----------------------:|:-----------------------:|:----------------------:|:----------:|
| Explanation | gradient-based | reinforcement learning | evaluationary algorithm | performance prediction | other types |


### 2019

|  Title  |   Venue  |   Type   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [Neural architecture search: A survey](http://www.jmlr.org/papers/volume20/18-598/18-598.pdf) | JMLR | Survey | - |
| [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) | ICLR | G | [github](https://github.com/quark0/darts) |
| [ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://openreview.net/pdf?id=HylVB3AqYm) | ICLR | RL/G | [github](https://github.com/MIT-HAN-LAB/ProxylessNAS) |
| [Graph HyperNetworks for Neural Architecture Search](https://arxiv.org/pdf/1810.05749.pdf) | ICLR | G | - |
| [Learnable Embedding Space for Efficient Neural Architecture Compression](https://openreview.net/forum?id=S1xLN3C9YX) | ICLR | Other | [github](https://github.com/Friedrich1006/ESNAC) |
| [Efficient Multi-Objective Neural Architecture Search via Lamarckian Evolution](https://arxiv.org/abs/1804.09081) | ICLR | EA | - |
| [SNAS: stochastic neural architecture search](https://openreview.net/pdf?id=rylqooRqK7) | ICLR | G | - |
| [Searching for A Robust Neural Architecture in Four GPU Hours](http://xuanyidong.com/publication/gradient-based-diff-sampler/) | CVPR | G | [github](https://github.com/D-X-Y/GDAS) |
| [ChamNet: Towards Efficient Network Design through Platform-Aware Model Adaptation](http://openaccess.thecvf.com/content_CVPR_2019/papers/Dai_ChamNet_Towards_Efficient_Network_Design_Through_Platform-Aware_Model_Adaptation_CVPR_2019_paper.pdf) | CVPR | - | - |
| [Partial Order Pruning: for Best Speed/Accuracy Trade-off in Neural Architecture Search](https://arxiv.org/pdf/1903.03777.pdf) | CVPR | EA | [github](https://github.com/lixincn2015/Partial-Order-Pruning) |
| [FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search](https://arxiv.org/abs/1812.03443) | CVPR | G | - | 
| [RENAS: Reinforced Evolutionary Neural Architecture Search	](https://arxiv.org/abs/1808.00193) | CVPR | G | - |
| [Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation](https://arxiv.org/pdf/1901.02985.pdf) | CVPR |  G | [GitHub](https://github.com/tensorflow/models/tree/master/research/deeplab) |
| [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/abs/1807.11626) | CVPR | RL | [Github](https://github.com/AnjieZheng/MnasNet-PyTorch) |
| [MFAS: Multimodal Fusion Architecture Search](https://arxiv.org/pdf/1903.06496.pdf) | CVPR | EA | - |
| [A Neurobiological Evaluation Metric for Neural Network Model Search](https://arxiv.org/pdf/1805.10726.pdf) | CVPR | Other | - |
| [Fast Neural Architecture Search of Compact Semantic Segmentation Models via Auxiliary Cells](https://arxiv.org/abs/1810.10804) | CVPR | RL | - |
| Customizable Architecture Search for Semantic Segmentation | CVPR | - | - |
| [Regularized Evolution for Image Classifier Architecture Search](https://arxiv.org/pdf/1802.01548.pdf) | AAAI | EA | - |
| AutoAugment: Learning Augmentation Policies from Data | CVPR | RL | - |
| Population Based Augmentation: Efficient Learning of Augmentation Policy Schedules | ICML | EA | - |
| [The Evolved Transformer](https://arxiv.org/pdf/1901.11117.pdf) | ICML | EA | [Github](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/evolved_transformer.py) |
| EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks | ICML | RL | - |
| [NAS-Bench-101: Towards Reproducible Neural Architecture Search](https://arxiv.org/abs/1902.09635) | ICML | Other | [Github](https://github.com/google-research/nasbench) | 

### 2018
|  Title  |   Venue  |   Type   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| Towards Automatically-Tuned Deep Neural Networks | BOOK | - | [GitHub](https://github.com/automl/Auto-PyTorch) |
| [Efficient Architecture Search by Network Transformation](https://arxiv.org/pdf/1707.04873.pdf) | AAAI | RL | [github](https://github.com/han-cai/EAS) |
| [Learning Transferable Architectures for Scalable Image Recognition](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zoph_Learning_Transferable_Architectures_CVPR_2018_paper.pdf) | CVPR | RL | [github](https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet) |
| [N2N learning: Network to Network Compression via Policy Gradient Reinforcement Learning](https://openreview.net/forum?id=B1hcZZ-AW) | ICLR | RL | - |
| [Practical Block-wise Neural Network Architecture Generation](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhong_Practical_Block-Wise_Neural_CVPR_2018_paper.pdf) | CVPR | RL | - | [Efficient Neural Architecture Search via Parameter Sharing](http://proceedings.mlr.press/v80/pham18a.html) | ICML | RL | [github](https://github.com/melodyguan/enas) |
| [Path-Level Network Transformation for Efficient Architecture Search](https://arxiv.org/abs/1806.02639) | ICML | RL | [github](https://github.com/han-cai/PathLevel-EAS) |
| [Hierarchical Representations for Efficient Architecture Search](https://openreview.net/forum?id=BJQRKzbA-) | ICLR | EA | - |
| [Understanding and Simplifying One-Shot Architecture Search](http://proceedings.mlr.press/v80/bender18a/bender18a.pdf) | ICML | G | - |
| [SMASH: One-Shot Model Architecture Search through HyperNetworks](https://arxiv.org/pdf/1708.05344.pdf) | ICLR | G | [github](https://github.com/ajbrock/SMASH) |
| [Neural Architecture Optimization](https://arxiv.org/pdf/1808.07233.pdf) | NeurIPS | G | [github](https://github.com/renqianluo/NAO) |
| [Searching for efficient multi-scale architectures for dense image prediction](https://papers.nips.cc/paper/8087-searching-for-efficient-multi-scale-architectures-for-dense-image-prediction.pdf) | NeurIPS | Other | - |
| [Progressive Neural Architecture Search](http://openaccess.thecvf.com/content_ECCV_2018/papers/Chenxi_Liu_Progressive_Neural_Architecture_ECCV_2018_paper.pdf) | ECCV | PD | [github](https://github.com/chenxi116/PNASNet) |
| [Neural Architecture Search with Bayesian Optimisation and Optimal Transport](https://arxiv.org/pdf/1802.07191.pdf) | NeurIPS | Other | [github](https://github.com/kirthevasank/nasbot) |
| [Differentiable Neural Network Architecture Search](https://openreview.net/pdf?id=BJ-MRKkwG) | ICLR-W | G | - |
| [Accelerating Neural Architecture Search using Performance Prediction](https://arxiv.org/abs/1705.10823) | ICLR-W | PD | - |



### 2017
|  Title  |   Venue  |   Type   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578) | ICLR | RL | - |
| [Designing Neural Network Architectures using Reinforcement Learning](https://openreview.net/pdf?id=S1c2cvqee) | ICLR | RL | - | [github](https://github.com/bowenbaker/metaqnn) |
| [Neural Optimizer Search with Reinforcement Learning](http://proceedings.mlr.press/v70/bello17a/bello17a.pdf) | ICML | RL | - | [Large-Scale Evolution of Image Classifiers](https://arxiv.org/pdf/1703.01041.pdf) | ICML | EA | - |
| [Learning Curve Prediction with Bayesian Neural Networks](http://ml.informatik.uni-freiburg.de/papers/17-ICLR-LCNet.pdf) | ICLR | PD | - |
| [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://arxiv.org/abs/1603.06560) | ICLR | PD | - |
| [Hyperparameter Optimization: A Spectral Approach](https://arxiv.org/abs/1706.00764) | NeurIPS-W | Other | [github](https://github.com/callowbird/Harmonica) |
| Learning to Compose Domain-Specific Transformations for Data Augmentation | NeurIPS | - | - |

### 2012-2016
|  Title  |   Venue  |   Type   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [Speeding up Automatic Hyperparameter Optimization of Deep Neural Networksby Extrapolation of Learning Curves](http://ml.informatik.uni-freiburg.de/papers/15-IJCAI-Extrapolation_of_Learning_Curves.pdf) | IJCAI | PD | [github](https://github.com/automl/pylearningcurvepredictor) |

### Blogs
- [What’s the deal with Neural Architecture Search?](https://determined.ai/blog/neural-architecture-search/)
- [Google Could AutoML](https://cloud.google.com/vision/automl/docs/beginners-guide)
- [PocketFlow](https://pocketflow.github.io/)
- [AutoML Challenge](http://automl.chalearn.org/)
- [AutoDL Challenge](https://autodl.chalearn.org/)


### arXiv
- Network Pruning via Transformable Architecture Search [[pdf]](https://arxiv.org/abs/1905.09717)
  - Xuanyi Dong, Yi Yang
- Population Based Training of Neural Networks [[pdf]](https://arxiv.org/abs/1711.09846)
  - Max Jaderberg, Valentin Dalibard, Simon Osindero, Wojciech M. Czarnecki, Jeff Donahue, Ali Razavi, Oriol Vinyals, Tim Green, Iain Dunning, Karen Simonyan, Chrisantha Fernando, Koray Kavukcuoglu. arXiv 1711
- NSGA-NET: A Multi-Objective Genetic Algorithm for Neural Architecture Search [[pdf]](https://arxiv.org/pdf/1810.03522.pdf)
  - Lu, Zhichao and Whalen, Ian and Boddeti, Vishnu and Dhebar, Yashesh and Deb, Kalyanmoy and Goodman, Erik and Banzhaf, Wolfgang, arXiv 1810
- Random Search and Reproducibility for Neural Architecture Search
  - Liam Li, Ameet Talwalkar. arXiv 1901
- Fast Neural Architecture Search of Compact Semantic Segmentation Models via Auxiliary Cells [[pdf]](https://arxiv.org/pdf/1810.10804.pdf)
  - Nekrasov, Vladimir and Chen, Hao and Shen, Chunhua and Reid, Ian. arXiv 1810
- Training Frankenstein’s Creature to Stack: HyperTree Architecture Search [[pdf]](https://arxiv.org/pdf/1810.11714.pdf)
  - Andrew Hundt, Varun Jain, Chris Paxton, Gregory D. Hager. arXiv 1810
- Fast, Accurate and Lightweight Super-Resolution with Neural Architecture Search [[pdf]](https://arxiv.org/pdf/1901.07261.pdf) [[code]](https://github.com/falsr/FALSR)



## Benchmark on ImageNet


| Architecture       | Top-1 (%) | Top-5 (%) | Params (M) | +x (M) | GPU | Days |
| ------------------ | --------- | --------- | ---------- | ------ | -   | -    |
| [Inception-v1](https://arxiv.org/pdf/1409.4842.pdf)       | 30.2      | 10.1      | 6.6        | 1448   | -   | -    |
| [MobileNet-v1](https://arxiv.org/abs/1704.04861)       | 29.4      | 10.5      | 4.2        | 569    | -   | -    |
| [ShuffleNet](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0642.pdf)         | 26.3      | -         | ~5         | 524    | -   | -    |
| [NASNet-A]((http://openaccess.thecvf.com/content_cvpr_2018/papers/Zoph_Learning_Transferable_Architectures_CVPR_2018_paper.pdf))           | 26.0      | 8.4       | 5.3        | 564    | 450 | 3-4  |
| NASNet-B           | 27.2      | 8.7       | 5.3        | 488    | 450 | 3-4  |
| NASNet-C           | 27.5      | 9.0       | 4.9        | 558    | 450 | 3-4  |
| [AmobebaNet-A](https://arxiv.org/pdf/1802.01548.pdf)       | 25.5      | 8.0       | 5.1        | 555    | 450 |  7   |
| AmobebaNet-B       | 26.0      | 8.5       | 5.3        | 555    | 450 |  7   |
| AmobebaNet-C       | 24.3      | 7.6       | 6.4        | 555    | 450 |  7   |
| [Progressive NAS](http://openaccess.thecvf.com/content_ECCV_2018/papers/Chenxi_Liu_Progressive_Neural_Architecture_ECCV_2018_paper.pdf)    | 25.8      | 8.1       | 5.1        | 588    | 100 | 1.5  |
| [DARTS-V2](https://arxiv.org/abs/1806.09055)           | 26.9      | 9.0       | 4.9        | 595    |  1  |  1   |
| [GDAS](http://xuanyidong.com/bibtex/Four-Hours-CVPR19.txt) | 26.0      | 8.5       | 5.3        | 581    |  1  |  0.21   |
