# Awesome NAS [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of neural architecture search and related resources. Inspired by [awesome-deep-vision](https://github.com/kjw0612/awesome-deep-vision), [awesome-adversarial-machine-learning](https://github.com/yenchenlin/awesome-adversarial-machine-learning), [awesome-deep-learning-papers](https://github.com/terryum/awesome-deep-learning-papers), and [awesome-architecture-search](https://github.com/markdtw/awesome-architecture-search).

Please feel free to [pull requests](https://github.com/D-X-Y/awesome-NAS/pulls) or [open an issue](https://github.com/D-X-Y/awesome-NAS/issues) to add papers.

## Table of Contents

- [Awesome Blogs](#awesome-blogs)

- [Neural Architecture Search](#NAS)
  - [2020 Venues](#2020)
  - [2019 Venues](#2019)
  - [2018 Venues](#2018)
  - [2017 Venues](#2017)
  - [Previous Venues](#2012-2016)
  - [arXiv](#arxiv)

## Awesome Blogs
- [AutoML info](http://automl.chalearn.org/) and [AutoML Freiburg-Hannover](https://www.automl.org/)
- [What’s the deal with Neural Architecture Search?](https://determined.ai/blog/neural-architecture-search/)
- [Google Could AutoML](https://cloud.google.com/vision/automl/docs/beginners-guide) and [PocketFlow](https://pocketflow.github.io/)
- [AutoML Challenge](http://automl.chalearn.org/) and [AutoDL Challenge](https://autodl.chalearn.org/)


## Neural Architecture Search (NAS)

|      Type   |        G       |                  RL    |            EA           |        PD              |    Other   |
|:------------|:--------------:|:----------------------:|:-----------------------:|:----------------------:|:----------:|
| Explanation | gradient-based | reinforcement learning | evaluationary algorithm | performance prediction | other types |

### 2020


|  Title  |   Venue  |   Type   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [When NAS Meets Robustness: In Search of Robust Architectures against Adversarial Attacks](https://arxiv.org/abs/1911.10695) | CVPR | G | [Github](https://github.com/gmh14/RobNets) |
| [Block-wisely Supervised Neural Architecture Search with Knowledge Distillation](https://www.xiaojun.ai/papers/CVPR2020_04676.pdf) | CVPR | G | [Github](https://github.com/changlin31/DNA) |
| [Overcoming Multi-Model Forgetting in One-Shot NAS with Diversity Maximization](https://www.xiaojun.ai/papers/cvpr-2020-zhang.pdf) | CVPR | G | [Github](https://github.com/MiaoZhang0525/NSAS_FOR_CVPR) |
| [Densely Connected Search Space for More Flexible Neural Architecture Search](https://arxiv.org/abs/1906.09607) | CVPR | G | [Github](https://github.com/JaminFong/DenseNAS) |
| [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070) | CVPR | RL | - |
| [NAS-BENCH-201: Extending the Scope of Reproducible Neural Architecture Search](https://openreview.net/forum?id=HJxyZkBKDr) | ICLR | - | [Github](https://github.com/D-X-Y/AutoDL-Projects) |
| [Understanding Architectures Learnt by Cell-based Neural Architecture Search](https://openreview.net/pdf?id=H1gDNyrKDS) | ICLR | G | [GitHub](https://github.com/automl/RobustDARTS) |
| [Evaluating The Search Phase of Neural Architecture Search](https://openreview.net/forum?id=H1loF2NFwr) | ICLR | - | |
| [AtomNAS: Fine-Grained End-to-End Neural Architecture Search](https://openreview.net/forum?id=BylQSxHFwr) | ICLR | | [GitHub](https://github.com/meijieru/AtomNAS) |
| [Fast Neural Network Adaptation via Parameter Remapping and Architecture Search](https://openreview.net/forum?id=rklTmyBKPH) | ICLR | - | [GitHub](https://github.com/JaminFong/FNA) |
| [Once for All: Train One Network and Specialize it for Efficient Deployment](https://openreview.net/forum?id=HylxE1HKwS) | ICLR | G | [GitHub](https://github.com/mit-han-lab/once-for-all) |
| Efficient Transformer for Mobile Applications | ICLR | - | - |
| PC-DARTS: Partial Channel Connections for Memory-Efficient Architecture Search | ICLR | G | [GitHub](https://github.com/yuhuixu1993/PC-DARTS) |
| Adversarial AutoAugment | ICLR | - | - |
| [NAS evaluation is frustratingly hard](https://arxiv.org/abs/1912.12522) | ICLR | - | [GitHub](https://github.com/antoyang/NAS-Benchmark) |
| [FasterSeg: Searching for Faster Real-time Semantic Segmentation](https://openreview.net/pdf?id=BJgqQ6NYvB) | ICLR | G | [GitHub](https://github.com/TAMU-VITA/FasterSeg) |
| [Computation Reallocation for Object Detection](https://openreview.net/forum?id=SkxLFaNKwB) | ICLR | - | - |
| Towards Fast Adaptation of Neural Architectures with Meta Learning | ICLR | - | - |
| AssembleNet: Searching for Multi-Stream Neural Connectivity in Video Architectures | ICLR | EA | - |
| How to Own the NAS in Your Spare Time | ICLR | - | - |

### 2019

|  Title  |   Venue  |   Type   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [DATA: Differentiable ArchiTecture Approximation](http://papers.nips.cc/paper/8374-data-differentiable-architecture-approximation) | NeurIPS | - | - |
| Random Search and Reproducibility for Neural Architecture Search | UAI | G | [GitHub](https://github.com/D-X-Y/NAS-Projects/blob/master/scripts-search/algos/RANDOM-NAS.sh) |
| [Improved Differentiable Architecture Search for Language Modeling and Named Entity Recognition](https://www.aclweb.org/anthology/D19-1367.pdf/) | EMNLP | G | - |
| [Continual and Multi-Task Architecture Search](https://www.aclweb.org/anthology/P19-1185.pdf) | ACL | RL | - |
| Progressive Differentiable Architecture Search: Bridging the Depth Gap Between Search and Evaluation | ICCV | - | - |
| Multinomial Distribution Learning for Effective Neural Architecture Search | ICCV | - | - |
| Searching for MobileNetV3 | ICCV | EA | - |
| [Multinomial Distribution Learning for Effective Neural Architecture Search](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zheng_Multinomial_Distribution_Learning_for_Effective_Neural_Architecture_Search_ICCV_2019_paper.pdf) | ICCV | - | [GitHub](https://github.com/tanglang96/MDENAS) |
| [Fast and Practical Neural Architecture Search](http://openaccess.thecvf.com/content_ICCV_2019/papers/Cui_Fast_and_Practical_Neural_Architecture_Search_ICCV_2019_paper.pdf) | ICCV | | |
| [Teacher Guided Architecture Search](http://openaccess.thecvf.com/content_ICCV_2019/papers/Bashivan_Teacher_Guided_Architecture_Search_ICCV_2019_paper.pdf) | ICCV | | - |
| [AutoDispNet: Improving Disparity Estimation With AutoML](http://openaccess.thecvf.com/content_ICCV_2019/papers/Saikia_AutoDispNet_Improving_Disparity_Estimation_With_AutoML_ICCV_2019_paper.pdf) | ICCV | G | - |
| [Resource Constrained Neural Network Architecture Search: Will a Submodularity Assumption Help?](http://openaccess.thecvf.com/content_ICCV_2019/papers/Xiong_Resource_Constrained_Neural_Network_Architecture_Search_Will_a_Submodularity_Assumption_ICCV_2019_paper.pdf) | ICCV | EA | - |
| [One-Shot Neural Architecture Search via Self-Evaluated Template Network](https://arxiv.org/abs/1910.05733) | ICCV | G | [Github](https://github.com/D-X-Y/NAS-Projects) |
| [Evolving Space-Time Neural Architectures for Videos](https://arxiv.org/abs/1811.10636) | ICCV | EA | [GitHub](https://sites.google.com/view/evanet-video) |
| [AutoGAN: Neural Architecture Search for Generative Adversarial Networks](https://arxiv.org/pdf/1908.03835.pdf) | ICCV | RL | [github](https://github.com/TAMU-VITA/AutoGAN) |
| [Discovering Neural Wirings](https://arxiv.org/pdf/1906.00586.pdf) | NeurIPS | G | [Github](https://github.com/allenai/dnw) |
| [Towards modular and programmable architecture search](https://arxiv.org/abs/1909.13404) | NeurIPS | [Other](https://github.com/D-X-Y/Awesome-NAS/issues/10) | [Github](https://github.com/negrinho/deep_architect) |
| [Network Pruning via Transformable Architecture Search](https://arxiv.org/abs/1905.09717) | NeurIPS | G | [Github](https://github.com/D-X-Y/NAS-Projects) |
| [Deep Active Learning with a NeuralArchitecture Search](https://arxiv.org/pdf/1811.07579.pdf) | NeurIPS | - | - |
| DetNAS: Backbone Search for ObjectDetection | NeurIPS | - | - |
| SpArSe: Sparse Architecture Search for CNNs on Resource-Constrained Microcontrollers | NeurIPS | - | - |
| [Efficient Forward Architecture Search ](https://arxiv.org/abs/1905.13360) | NeurIPS | G | [Github](https://github.com/microsoft/petridishnn) |
| Efficient Neural ArchitectureTransformation Search in Channel-Level for Object Detection | NeurIPS | G | - |
| XNAS: Neural Architecture Search with Expert Advice | NeurIPS | G | - |
| [Neural architecture search: A survey](http://www.jmlr.org/papers/volume20/18-598/18-598.pdf) | JMLR | Survey | - |
| [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) | ICLR | G | [github](https://github.com/quark0/darts) |
| [ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://openreview.net/pdf?id=HylVB3AqYm) | ICLR | RL/G | [github](https://github.com/MIT-HAN-LAB/ProxylessNAS) |
| [Graph HyperNetworks for Neural Architecture Search](https://arxiv.org/pdf/1810.05749.pdf) | ICLR | G | - |
| [Learnable Embedding Space for Efficient Neural Architecture Compression](https://openreview.net/forum?id=S1xLN3C9YX) | ICLR | Other | [github](https://github.com/Friedrich1006/ESNAC) |
| [Efficient Multi-Objective Neural Architecture Search via Lamarckian Evolution](https://arxiv.org/abs/1804.09081) | ICLR | EA | - |
| [SNAS: stochastic neural architecture search](https://openreview.net/pdf?id=rylqooRqK7) | ICLR | G | - |
| [Searching for A Robust Neural Architecture in Four GPU Hours](http://xuanyidong.com/publication/gradient-based-diff-sampler/) | CVPR | G | [Github](https://github.com/D-X-Y/NAS-Projects) |
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
| [A Flexible Approach to Automated RNN Architecture Generation](https://openreview.net/forum?id=SkOb1Fl0Z) | ICLR | RL/PD | - |
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




### arXiv
|  Title  |   Date  |   Type   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [Population Based Training of Neural Networks](https://arxiv.org/abs/1711.09846) | 2017.11 | EA | - |
| [NSGA-NET: A Multi-Objective Genetic Algorithm for Neural Architecture Search](https://arxiv.org/pdf/1810.03522.pdf) | 2018.10 | EA | - |
| [Training Frankenstein’s Creature to Stack: HyperTree Architecture Search](https://arxiv.org/pdf/1810.11714.pdf) | 2018.10 | G | - |
| [Fast, Accurate and Lightweight Super-Resolution with Neural Architecture Search](https://arxiv.org/pdf/1901.07261.pdf) | 2019.01 | G | [github](https://github.com/falsr/FALSR) |
