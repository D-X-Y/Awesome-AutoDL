# Awesome NAS [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of neural architecture search and related resources. Inspired by [awesome-deep-vision](https://github.com/kjw0612/awesome-deep-vision), [awesome-adversarial-machine-learning](https://github.com/yenchenlin/awesome-adversarial-machine-learning), [awesome-deep-learning-papers](https://github.com/terryum/awesome-deep-learning-papers), and [awesome-architecture-search](https://github.com/markdtw/awesome-architecture-search).

Please feel free to [pull requests](https://github.com/D-X-Y/awesome-NAS/pulls) or [open an issue](https://github.com/D-X-Y/awesome-NAS/issues) to add papers.

## Table of Contents

- [Neural Architecture Search](#NAS)
  - [Blogs](#blogs)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Evolutionary Algorithm](#evolutionary-algorithm)
  - [Gradient-based](#gradient-based-approach)
  - [Performance Prediction-based](#performance-prediction-based)
  - [Multi-Objective NAS](#multi-objective-nas)
  - [Others](#others)
- [NAS for Application](#NAS-for-Application)
- [Survey](#Survey)
- [Benchmark on ImageNet](#benchmark-on-imagenet)


## NAS

### Blogs
- [What’s the deal with Neural Architecture Search?](https://determined.ai/blog/neural-architecture-search/)
- [PocketFlow](https://pocketflow.github.io/)
- [AutoDL](http://automl.chalearn.org/)

### Reinforcement Learning
- Neural Architecture Search with Reinforcement Learning [[pdf]](https://arxiv.org/abs/1611.01578)
  - Barret Zoph and Quoc V. Le. ICLR 2017
- Designing Neural Network Architectures using Reinforcement Learning [[pdf]](https://openreview.net/pdf?id=S1c2cvqee)
  - Baker, Bowen and Gupta, Otkrist and Naik, Nikhil and Raskar, Ramesh. ICLR 2017
- Neural Optimizer Search with Reinforcement Learning [[pdf]](http://proceedings.mlr.press/v70/bello17a/bello17a.pdf)
  - Bello, Irwan and Zoph, Barret and Vasudevan, Vijay and Le, Quoc V. ICML 2017
- Efficient Architecture Search by Network Transformation [[pdf]](https://arxiv.org/pdf/1707.04873.pdf)
  - Han Cai, Tianyao Chen, Weinan Zhang, Yong Yu, Jun Wang. AAAI 2018
- Learning Transferable Architectures for Scalable Image Recognition [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zoph_Learning_Transferable_Architectures_CVPR_2018_paper.pdf)
  - Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le. CVPR 2018
- Practical Block-wise Neural Network Architecture Generation [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhong_Practical_Block-Wise_Neural_CVPR_2018_paper.pdf)
  - Zhong, Zhao and Yan, Junjie and Wu, Wei and Shao, Jing and Liu, Cheng-Lin. CVPR 2018
- Efficient Neural Architecture Search via Parameter Sharing [[pdf]](http://proceedings.mlr.press/v80/pham18a.html)
  - Pham, Hieu and Guan, Melody Y and Zoph, Barret and Le, Quoc V and Dean, Jeff. ICML 2018
- MnasNet: Platform-Aware Neural Architecture Search for Mobile [[pdf]](https://arxiv.org/pdf/1807.11626.pdf) [[code]](https://github.com/AnjieZheng/MnasNet-PyTorch)
  - Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Quoc V. Le. arXiv 2018.07
- ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
  - Cai, Han and Zhu, Ligeng and Han, Song. ICLR 2019
- Path-Level Network Transformation for Efficient Architecture Search
  - Cai, Han and Yang, Jiacheng and Zhang, Weinan and Han, Song and Yu, Yong. ICML 2018



### Evolutionary Algorithm
- Population Based Training of Neural Networks [[pdf]](https://arxiv.org/abs/1711.09846)
  - Max Jaderberg, Valentin Dalibard, Simon Osindero, Wojciech M. Czarnecki, Jeff Donahue, Ali Razavi, Oriol Vinyals, Tim Green, Iain Dunning, Karen Simonyan, Chrisantha Fernando, Koray Kavukcuoglu. arXiv 1711
- Large-Scale Evolution of Image Classifiers [[pdf]](https://arxiv.org/pdf/1703.01041.pdf)
  - Real, Esteban and Moore, Sherry and Selle, Andrew and Saxena, Saurabh and Suematsu, Yutaka Leon and Tan, Jie and Le, Quoc and Kurakin, Alex. ICML 2017
- Hierarchical Representations for Efficient Architecture Search [[pdf]](https://openreview.net/forum?id=BJQRKzbA-)
  - Hanxiao Liu, Karen Simonyan, Oriol Vinyals, Chrisantha Fernando, Koray Kavukcuoglu. ICLR 2018
- Regularized Evolution for Image Classifier Architecture Search [[pdf]](https://arxiv.org/pdf/1802.01548.pdf)
  - Real, Esteban and Aggarwal, Alok and Huang, Yanping and Le, Quoc V. AAAI 2019
- Partial Order Pruning: for Best Speed/Accuracy Trade-off in Neural Architecture Search [[pdf]](https://arxiv.org/pdf/1903.03777.pdf)
  - Xin Li, Yiming Zhou, Zheng Pan, Jiashi Feng. CVPR 2019



### Gradient-based Approach
- Differentiable Neural Network Architecture Search [[pdf]](https://openreview.net/pdf?id=BJ-MRKkwG)
  - Richard Shin, Charles Packer, Dawn Song, ICLR 2018 Workshop
- Understanding and Simplifying One-Shot Architecture Search [[pdf]](http://proceedings.mlr.press/v80/bender18a/bender18a.pdf)
  - Bender, Gabriel and Kindermans, Pieter-Jan and Zoph, Barret and Vasudevan, Vijay and Le, Quoc. ICML 2018
- SMASH: One-Shot Model Architecture Search through HyperNetworks [[pdf]](https://arxiv.org/pdf/1708.05344.pdf)
  - Brock, Andrew and Lim, Theodore and Ritchie, James M and Weston, Nick. ICLR 2018
- Neural Architecture Optimization [[pdf]](https://arxiv.org/pdf/1808.07233.pdf) [[code]](https://github.com/renqianluo/NAO)
  - Luo, Renqian and Tian, Fei and Qin, Tao and Liu, Tie-Yan. NIPS 2018
- DARTS: Differentiable Architecture Search [[pdf]](https://arxiv.org/abs/1806.09055) [[code]](https://github.com/quark0/darts)
  - Hanxiao Liu, Karen Simonyan, Yiming Yang. ICLR 2019
- Graph HyperNetworks for Neural Architecture Search [[pdf]](https://arxiv.org/pdf/1810.05749.pdf)
  - Chris Zhang, Mengye Ren, Raquel Urtasun. ICLR 2019
- SNAS: stochastic neural architecture search
  - Sirui Xie, Hehui Zheng, Chunxiao Liu, Liang Lin. ICLR 2019
- ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
  - Cai, Han and Zhu, Ligeng and Han, Song. ICLR 2019
- Searching for A Robust Neural Architecture in Four GPU Hours
  - Dong, Xuanyi and Yang, Yi. CVPR 2019
- Random Search and Reproducibility for Neural Architecture Search
  - Liam Li, Ameet Talwalkar. arXiv 1901


### Performance Prediction-based
- Speeding up Automatic Hyperparameter Optimization of Deep Neural Networksby Extrapolation of Learning Curves [[pdf]](http://ml.informatik.uni-freiburg.de/papers/15-IJCAI-Extrapolation_of_Learning_Curves.pdf) [[code]](https://github.com/automl/pylearningcurvepredictor)
  - Tobias Domhan, Jost Tobias Springenberg, Frank Hutter. IJCAI 2015
- Learning Curve Prediction with Bayesian Neural Networks [[pdf]](http://ml.informatik.uni-freiburg.de/papers/17-ICLR-LCNet.pdf)
  - Aaron Klein, Stefan Falkner, Jost Tobias Springenberg, Frank Hutter. ICLR 2017
- Progressive Neural Architecture Search [[pdf]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Chenxi_Liu_Progressive_Neural_Architecture_ECCV_2018_paper.pdf) [[code]](https://github.com/chenxi116/PNASNet)
  - Chenxi Liu, Barret Zoph, Jonathon Shlens, Wei Hua, Li-Jia Li, Li Fei-Fei, Alan Yuille, Jonathan Huang, Kevin Murphy. ECCV 2018
- Accelerating Neural Architecture Search using Performance Prediction [[pdf]](https://arxiv.org/abs/1705.10823)
  - Bowen Baker, Otkrist Gupta, Ramesh Raskar, Nikhil Naik. ICLR 2018 Workshop

### Multi-Objective NAS
- NSGA-NET: A Multi-Objective Genetic Algorithm for Neural Architecture Search [[pdf]](https://arxiv.org/pdf/1810.03522.pdf)
  - Lu, Zhichao and Whalen, Ian and Boddeti, Vishnu and Dhebar, Yashesh and Deb, Kalyanmoy and Goodman, Erik and Banzhaf, Wolfgang, arXiv 1810
- Efficient Multi-Objective Neural Architecture Search via Lamarckian Evolution
  - Thomas Elsken, Jan Hendrik Metzen, Frank Hutter. ICLR 2019


### Others
- Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization [[pdf]](https://arxiv.org/abs/1603.06560)
  - Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin Rostamizadeh, Ameet Talwalkar. ICLR 2017
- Hyperparameter Optimization: A Spectral Approach [[pdf]](https://arxiv.org/abs/1706.00764) [[code]](https://github.com/callowbird/Harmonica)
  - Elad Hazan, Adam Klivans, Yang Yuan. NIPS 2017 Workshop
- Neural Architecture Search with Bayesian Optimisation and Optimal Transport [[pdf]](https://arxiv.org/pdf/1802.07191.pdf)
  - Kandasamy, Kirthevasan and Neiswanger, Willie and Schneider, Jeff and Poczos, Barnabas and Xing, Eric. NIPS 2018

## NAS for Application
- Fast Neural Architecture Search of Compact Semantic Segmentation Models via Auxiliary Cells [[pdf]](https://arxiv.org/pdf/1810.10804.pdf)
  - Nekrasov, Vladimir and Chen, Hao and Shen, Chunhua and Reid, Ian. arXiv 1810
- Training Frankenstein’s Creature to Stack: HyperTree Architecture Search [[pdf]](https://arxiv.org/pdf/1810.11714.pdf)
  - Andrew Hundt, Varun Jain, Chris Paxton, Gregory D. Hager. arXiv 1810
- Searching for efficient multi-scale architectures for dense image prediction [[pdf]](https://arxiv.org/pdf/1809.04184.pdf)
  - Chen, Liang-Chieh and Collins, Maxwell D and Zhu, Yukun and Papandreou, George and Zoph, Barret and Schroff, Florian and Adam, Hartwig and Shlens, Jonathon. NIPS 2018
- Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation [[pdf]](https://arxiv.org/abs/1901.02985)
  - arXiv 1901
- Fast, Accurate and Lightweight Super-Resolution with Neural Architecture Search [[pdf]](https://arxiv.org/pdf/1901.07261.pdf) [[code]](https://github.com/falsr/FALSR)

## Survey
- Neural Architecture Search: A Survey
  - Thomas Elsken, Jan Hendrik Metzen, Frank Hutter. arXiv 1808
- Taking Human out of Learning Applications: A Survey on Automated Machine Learning [[pdf]](https://arxiv.org/abs/1810.13306)
  - Yao Quanming, Wang Mengshuo, Jair Escalante Hugo, Guyon Isabelle, Hu Yi-Qi, Li Yu-Feng, Tu Wei-Wei, Yang Qiang, Yu Yang. arXiv 1810
 
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
| [DARTS-V2](https://arxiv.org/abs/1806.09055)           | 26.9      | 9.0       | 4.9        | 595    |  1  |  4   |
