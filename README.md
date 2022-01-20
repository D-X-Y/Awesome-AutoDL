<font size=6><center><big><b> Awesome AutoDL [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) </b></big></center></font>

A curated list of automated deep learning related resources. Inspired by [awesome-deep-vision](https://github.com/kjw0612/awesome-deep-vision), [awesome-adversarial-machine-learning](https://github.com/yenchenlin/awesome-adversarial-machine-learning), [awesome-deep-learning-papers](https://github.com/terryum/awesome-deep-learning-papers), and [awesome-architecture-search](https://github.com/markdtw/awesome-architecture-search).

Please feel free to [pull requests](https://github.com/D-X-Y/Awesome-AutoDL/pulls) or [open an issue](https://github.com/D-X-Y/Awesome-AutoDL/issues) to add papers.

---

<font size=5><center><b> Table of Contents </b> </center></font>

- [Awesome Blogs](#awesome-blogs)
- [Awesome AutoDL Libraies](#awesome-autodl-libraies)
- [Awesome Benchmarks](#awesome-benchmarks)
- [Deep Learning-based NAS and HPO](#deep-learning-based-nas-and-hpo)
  - [2021 Venues](#2021-venues)
  - [2020 Venues](#2020-venues)
  - [2019 Venues](#2019-venues)
  - [2018 Venues](#2018-venues)
  - [2017 Venues](#2017-venues)
  - [Previous Venues](#previous-venues)
  - [arXiv](#arxiv)
- [Awesome Surveys](#awesome-surveys)

---

# Awesome Blogs

- [AutoML info](http://automl.chalearn.org/) and [AutoML Freiburg-Hannover](https://www.automl.org/)
- [What’s the deal with Neural Architecture Search?](https://determined.ai/blog/neural-architecture-search/)
- [Google Could AutoML](https://cloud.google.com/vision/automl/docs/beginners-guide) and [PocketFlow](https://pocketflow.github.io/)
- [AutoML Challenge](http://automl.chalearn.org/) and [AutoDL Challenge](https://autodl.chalearn.org/)
- [In Defense of Weight-sharing for Neural Architecture Search: an optimization perspective](https://determined.ai/blog/ws-optimization-for-nas/)

# Awesome AutoDL Libraies

- [PyGlove](https://proceedings.neurips.cc/paper/2020/file/012a91467f210472fab4e11359bbfef6-Paper.pdf)
- [NASLib](https://github.com/automl/NASLib)
- [Keras Tuner](https://keras-team.github.io/keras-tuner/)
- [NNI](https://github.com/microsoft/nni)
- [AutoGluon](https://autogluon.mxnet.io/)
- [Auto-PyTorch](https://github.com/automl/Auto-PyTorch)
- [AutoDL-Projects](https://github.com/D-X-Y/AutoDL-Projects)
- [aw_nas](https://github.com/walkerning/aw_nas)
- [Determined](https://github.com/determined-ai/determined)
- [TPOT](https://github.com/EpistasisLab/tpot)

# Awesome Benchmarks

| Title | Venue | Code |
|:--------|:--------:|:--------:|
| [NAS-Bench-101: Towards Reproducible Neural Architecture Search](https://arxiv.org/pdf/1902.09635.pdf) | ICML 2019 | [GitHub](https://github.com/google-research/nasbench) |
| [NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search](https://openreview.net/forum?id=HJxyZkBKDr) | ICLR 2020 | [Github](https://github.com/D-X-Y/NAS-Bench-201) |
| [NAS-Bench-301 and the Case for Surrogate Benchmarks for Neural Architecture Search](https://arxiv.org/abs/2008.09777) | arXiv 2020 | [GitHub](https://github.com/automl/nasbench301) |
| [NAS-Bench-1Shot1: Benchmarking and Dissecting One-shot Neural Architecture Search](https://arxiv.org/abs/2001.10422) | ICLR 2020 | [GitHub](https://github.com/automl/nasbench-1shot1) |
| [NATS-Bench: Benchmarking NAS Algorithms for Architecture Topology and Size](https://arxiv.org/abs/2009.00437) | TPAMI 2021 | [GitHub](https://github.com/D-X-Y/NATS-Bench)
| [NAS-Bench-ASR: Reproducible Neural Architecture Search for Speech Recognition](https://openreview.net/forum?id=CU0APx9LMaL) | ICLR 2021 | - |
| [HW-NAS-Bench: Hardware-Aware Neural Architecture Search Benchmark](https://openreview.net/pdf?id=_0kaDkv3dVf) | ICLR 2021 | [GitHub](https://github.com/RICE-EIC/HW-NAS-Bench) |
| [NAS-Bench-NLP: Neural Architecture Search Benchmark for Natural Language Processing](https://arxiv.org/pdf/2006.07116.pdf) | arXiv 2020 | [GitHub](https://github.com/fmsnew/nas-bench-nlp-release) |
| [NAS-Bench-x11 and the Power of Learning Curves](https://arxiv.org/pdf/2111.03602.pdf) | NeurIPS 2021 | [GitHub](https://github.com/automl/nas-bench-x11) |

# Deep Learning-based NAS and HPO

|      Type   |        G       |                  RL    |            EA           |        PD              |    Other   |
|:------------|:--------------:|:----------------------:|:-----------------------:|:----------------------:|:----------:|
| Explanation | gradient-based | reinforcement learning | evolutionary algorithm | performance prediction | other types |

## 2021 Venues

|  Title  |   Venue  |   Type   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [CATE: Computation-aware Neural Architecture Encoding with Transformers](https://arxiv.org/pdf/2102.07108.pdf) | ICML | O | [GitHub](https://github.com/MSU-MLSys-Lab/CATE) |
| [Searching by Generating: Flexible and Efficient One-Shot NAS with Architecture Generator](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_Searching_by_Generating_Flexible_and_Efficient_One-Shot_NAS_With_Architecture_CVPR_2021_paper.pdf) | CVPR | G | [Github](https://github.com/eric8607242/SGNAS) |
| [Zen-NAS: A Zero-Shot NAS for High-Performance Deep Image Recognition](https://arxiv.org/abs/2102.01063) | ICCV | EA | [Github](https://github.com/idstcv/ZenNAS) |
| [AutoFormer: Searching Transformers for Visual Recognition](https://arxiv.org/pdf/2107.00651.pdf) |ICCV | EA | [GitHub](https://github.com/microsoft/AutoML)
| [LightTrack: Finding Lightweight Neural Networks for Object Tracking via One-Shot Architecture Search](https://arxiv.org/abs/2104.14545) | CVPR | EA | [GitHub](https://github.com/researchmm/LightTrack) |
| [One-Shot Neural Ensemble Architecture Search by Diversity-Guided Search Space Shrinking](https://arxiv.org/abs/2104.00597) | CVPR | EA | [GitHub](https://github.com/researchmm/NEAS) |
| [DARTS-: Robustly Stepping out of Performance Collapse Without Indicators](https://openreview.net/pdf?id=KLH36ELmwIB) | ICLR | G | [GitHub](https://github.com/Meituan-AutoML/DARTS-) |
| [Zero-Cost Proxies for Lightweight NAS](https://openreview.net/pdf?id=0cmMMy8J5q) | ICLR | O | [GitHub](https://github.com/SamsungLabs/zero-cost-nas) |
| [Neural Architecture Search on ImageNet in Four GPU Hours: A Theoretically Inspired Perspective](https://openreview.net/forum?id=Cnon5ezMHtu) | ICLR | - | [GitHub](https://github.com/VITA-Group/TENAS) |
| [DrNAS: Dirichlet Neural Architecture Search](https://openreview.net/forum?id=9FWas6YbmB3) | ICLR | G | [GitHub](https://github.com/xiangning-chen/DrNAS) |
| [Rethinking Architecture Selection in Differentiable NAS](https://openreview.net/forum?id=PKubaeJkw3) | ICLR | O | [GitHub](https://github.com/ruocwang/darts-pt) |
| [Evolving Reinforcement Learning Algorithms](https://openreview.net/forum?id=0XXpJ4OtjW) | ICLR | EA | [GitHub](https://github.com/jcoreyes/evolvingrl) |
| [AutoHAS: Differentiable Hyper-parameter and Architecture Search](https://arxiv.org/pdf/2006.03656.pdf) | ICLR-W | G | - |
| [FBNetV3: Joint Architecture-Recipe Search using Neural Acquisition Function](https://arxiv.org/abs/2006.02049) | CVPR | PD | [github](https://github.com/facebookresearch/mobile-vision/blob/main/mobile_cv/arch/fbnet_v2/fbnet_modeldef_cls_fbnetv3.py) |

## 2020 Venues

|  Title  |   Venue  |   Type   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [Cream of the Crop: Distilling Prioritized Paths For One-Shot Neural Architecture Search](https://papers.nips.cc/paper/2020/file/d072677d210ac4c03ba046120f0802ec-Paper.pdf) | NeurIPS | - | [GitHub](https://github.com/microsoft/Cream) |
| [PyGlove: Symbolic Programming for Automated Machine Learning](https://proceedings.neurips.cc/paper/2020/file/012a91467f210472fab4e11359bbfef6-Paper.pdf) | NeurIPS | library | - |
| [Does Unsupervised Architecture Representation Learning Help Neural Architecture Search](https://arxiv.org/abs/2006.06936) | NeurIPS | PD | [GitHub](https://github.com/MSU-MLSys-Lab/arch2vec) |
| [RandAugment: Practical Automated Data Augmentation with a Reduced Search Space](https://arxiv.org/abs/1909.13719) | NeurIPS | | [GitHub](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) |
| [Delta-STN: Efficient Bilevel Optimization for Neural Networks using Structured Response Jacobians](https://arxiv.org/pdf/2010.13514.pdf) | NeurIPS | G | [GitHub](https://github.com/pomonam/Self-Tuning-Networks) |
| [A Study on Encodings for Neural Architecture Search](https://arxiv.org/abs/2007.04965) | NeurIPS | | [GitHub](https://github.com/naszilla/naszilla) |
| [AutoBSS: An Efficient Algorithm for Block Stacking Style Search](https://proceedings.neurips.cc/paper/2020/file/747d3443e319a22747fbb873e8b2f9f2-Paper.pdf) | NeurIPS | | |
| [Bridging the Gap between Sample-based and One-shot Neural Architecture Search with BONAS](https://proceedings.neurips.cc/paper/2020/file/13d4635deccc230c944e4ff6e03404b5-Paper.pdf) | NeurIPS | G | [GitHub](https://github.com/haolibai/APS-channel-search) |
| [Interstellar: Searching Recurrent Architecture for Knowledge Graph Embedding](https://proceedings.neurips.cc/paper/2020/file/722caafb4825ef5d8670710fa29087cf-Paper.pdf) | NeurIPS | | |
| [Revisiting Parameter Sharing for Automatic Neural Channel Number Search](https://proceedings.neurips.cc/paper/2020/file/42cd63cb189c30ed03e42ce2c069566c-Paper.pdf) | NeurIPS | | |
| [Learning Search Space Partition for Black-box Optimization using Monte Carlo Tree Search](https://arxiv.org/pdf/2007.00708.pdf) | NeurIPS | MCTS | [GitHub](https://github.com/facebookresearch/LaMCTS) |
| [Neural Architecture Search using Deep Neural Networks and Monte Carlo Tree Search](https://arxiv.org/abs/1805.07440) | AAAI | MCTS | [GitHub](https://github.com/linnanwang/AlphaX-NASBench101) |
| [Representation Sharing for Fast Object Detector Search and Beyond](https://arxiv.org/pdf/2007.12075v4.pdf) | ECCV | G | [GitHub](https://github.com/msight-tech/research-fad) |
| [Are Labels Necessary for Neural Architecture Search?](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490766.pdf) | ECCV | G | - |
| [Single Path One-Shot Neural Architecture Search with Uniform Sampling](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610528.pdf) | ECCV | EA | - |
| [Neural Predictor for Neural Architecture Search](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740647.pdf) | ECCV | O | - |
| [BigNAS: Scaling Up Neural Architecture Search with Big Single-Stage Models](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520681.pdf) | ECCV | G | - |
| [BATS: Binary ArchitecTure Search](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680307.pdf) | ECCV | - | - |
| [AttentionNAS: Spatiotemporal Attention Cell Search for Video Classification](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530443.pdf) | ECCV | - | - |
| [Search What You Want: Barrier Panelty NAS for Mixed Precision Quantization](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540001.pdf) | ECCV | - | - |
| [Angle-based Search Space Shrinking for Neural Architecture Search](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630426.pdf) | ECCV | - | - |
| [Anti-Bandit Neural Architecture Search for Model Defense](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580069.pdf) | ECCV | - | - |
| [TF-NAS: Rethinking Three Search Freedoms of Latency-Constrained Differentiable Neural Architecture Search](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600120.pdf) | ECCV | G | [GitHub](https://github.com/AberHu/TF-NAS) |
| [Fair DARTS: Eliminating Unfair Advantages in Differentiable Architecture Search](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600460.pdf) | ECCV | G | [GitHub](https://github.com/xiaomi-automl/FairDARTS) |
| [Off-Policy Reinforcement Learning for Efficient and Effective GAN Architecture Search](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520171.pdf) | ECCV | RL | - |
| [DA-NAS: Data Adapted Pruning for Efficient Neural Architecture Search](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720579.pdf) | ECCV | G | - |
| [Optimizing Millions of Hyperparameters by Implicit Differentiation](https://arxiv.org/abs/1911.02590) | AISTATS | G | - |
| [Evolving Machine Learning Algorithms From Scratch](https://arxiv.org/pdf/2003.03384.pdf) | ICML | EA | - |
| [Stabilizing Differentiable Architecture Search via Perturbation-based Regularization](https://arxiv.org/abs/2002.05283) | ICML | G | [GitHub](https://github.com/xiangning-chen/SmoothDARTS) |
| [NADS: Neural Architecture Distribution Search for Uncertainty Awareness](https://arxiv.org/pdf/2006.06646.pdf) | ICML | - | - |
| [Generative Teaching Networks: Accelerating Neural Architecture Search by Learning to Generate Synthetic Training Data](https://arxiv.org/abs/1912.07768) | ICML | - | - |
| Neural Architecture Search in a Proxy Validation Loss Landscape | ICML | - | - |
| [Hit-Detector: Hierarchical Trinity Architecture Search for Object Detection](https://arxiv.org/pdf/2003.11818v1.pdf) | CVPR | - | [GitHub](https://github.com/ggjy/HitDet.pytorch) |
| [Designing Network Design Spaces](https://arxiv.org/pdf/2003.13678.pdf) | CVPR | - | [GitHub](https://github.com/facebookresearch/pycls) |
| [UNAS: Differentiable Architecture Search Meets Reinforcement Learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/Vahdat_UNAS_Differentiable_Architecture_Search_Meets_Reinforcement_Learning_CVPR_2020_paper.pdf) | CVPR | G/RL | [GitHub](https://github.com/NVlabs/unas) |
| [MiLeNAS: Efficient Neural Architecture Search via Mixed-Level Reformulation](https://arxiv.org/pdf/2003.12238.pdf) | CVPR | G | [GitHub](https://github.com/chaoyanghe/MiLeNAS) |
| [A Semi-Supervised Assessor of Neural Architectures](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tang_A_Semi-Supervised_Assessor_of_Neural_Architectures_CVPR_2020_paper.pdf) | CVPR | PD | - |
| [Binarizing MobileNet via Evolution-based Searching](https://arxiv.org/abs/2005.06305) | CVPR | EA | - |
| [Rethinking Performance Estimation in Neural Architecture Search](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zheng_Rethinking_Performance_Estimation_in_Neural_Architecture_Search_CVPR_2020_paper.pdf) | CVPR | - | [GitHub](https://github.com/zhengxiawu/rethinking_performance_estimation_in_NAS) |
| [APQ: Joint Search for Network Architecture, Pruning and Quantization Policy](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_APQ_Joint_Search_for_Network_Architecture_Pruning_and_Quantization_Policy_CVPR_2020_paper.pdf) | CVPR | G | [GitHub](https://github.com/mit-han-lab/apq) |
| [SGAS: Sequential Greedy Architecture Search](http://openaccess.thecvf.com/content_CVPR_2020/papers/Li_SGAS_Sequential_Greedy_Architecture_Search_CVPR_2020_paper.pdf) | CVPR | G | [Github](https://github.com/lightaime/sgas) |
| [Can Weight Sharing Outperform Random Architecture Search? An Investigation With TuNAS](http://openaccess.thecvf.com/content_CVPR_2020/papers/Bender_Can_Weight_Sharing_Outperform_Random_Architecture_Search_An_Investigation_With_CVPR_2020_paper.pdf) | CVPR | RL | - |
| [FBNetV2: Differentiable Neural Architecture Search for Spatial and Channel Dimensions](https://arxiv.org/abs/2004.05565) | CVPR | G | [Github](https://github.com/facebookresearch/mobile-vision) |
| [AdversarialNAS: Adversarial Neural Architecture Search for GANs](https://arxiv.org/pdf/1912.02037.pdf) | CVPR | G | [Github](https://github.com/chengaopro/AdversarialNAS) |
| [When NAS Meets Robustness: In Search of Robust Architectures against Adversarial Attacks](https://arxiv.org/abs/1911.10695) | CVPR | G | [Github](https://github.com/gmh14/RobNets) |
| [Block-wisely Supervised Neural Architecture Search with Knowledge Distillation](https://www.xiaojun.ai/papers/CVPR2020_04676.pdf) | CVPR | G | [Github](https://github.com/changlin31/DNA) |
| [Overcoming Multi-Model Forgetting in One-Shot NAS with Diversity Maximization](https://www.xiaojun.ai/papers/cvpr-2020-zhang.pdf) | CVPR | G | [Github](https://github.com/MiaoZhang0525/NSAS_FOR_CVPR) |
| [Densely Connected Search Space for More Flexible Neural Architecture Search](https://arxiv.org/abs/1906.09607) | CVPR | G | [Github](https://github.com/JaminFong/DenseNAS) |
| [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070) | CVPR | RL | - |
| [NAS-BENCH-201: Extending the Scope of Reproducible Neural Architecture Search](https://openreview.net/forum?id=HJxyZkBKDr) | ICLR | - | [Github](https://github.com/D-X-Y/AutoDL-Projects) |
| [Understanding Architectures Learnt by Cell-based Neural Architecture Search](https://openreview.net/forum?id=BJxH22EKPS) | ICLR | G | [GitHub](https://github.com/shuyao95/Understanding-NAS) |
| [Evaluating The Search Phase of Neural Architecture Search](https://openreview.net/forum?id=H1loF2NFwr) | ICLR | - | |
| [AtomNAS: Fine-Grained End-to-End Neural Architecture Search](https://openreview.net/forum?id=BylQSxHFwr) | ICLR | | [GitHub](https://github.com/meijieru/AtomNAS) |
| [Fast Neural Network Adaptation via Parameter Remapping and Architecture Search](https://openreview.net/forum?id=rklTmyBKPH) | ICLR | - | [GitHub](https://github.com/JaminFong/FNA) |
| [Once for All: Train One Network and Specialize it for Efficient Deployment](https://openreview.net/forum?id=HylxE1HKwS) | ICLR | G | [GitHub](https://github.com/mit-han-lab/once-for-all) |
| Efficient Transformer for Mobile Applications | ICLR | - | - |
| [PC-DARTS: Partial Channel Connections for Memory-Efficient Architecture Search](https://arxiv.org/pdf/1907.05737v4.pdf) | ICLR | G | [GitHub](https://github.com/yuhuixu1993/PC-DARTS) |
| Adversarial AutoAugment | ICLR | - | - |
| [NAS evaluation is frustratingly hard](https://arxiv.org/abs/1912.12522) | ICLR | - | [GitHub](https://github.com/antoyang/NAS-Benchmark) |
| [FasterSeg: Searching for Faster Real-time Semantic Segmentation](https://openreview.net/pdf?id=BJgqQ6NYvB) | ICLR | G | [GitHub](https://github.com/TAMU-VITA/FasterSeg) |
| [Computation Reallocation for Object Detection](https://openreview.net/forum?id=SkxLFaNKwB) | ICLR | - | - |
| [Towards Fast Adaptation of Neural Architectures with Meta Learning](https://openreview.net/pdf?id=r1eowANFvr) | ICLR | - | [GitHub](https://github.com/dongzelian/T-NAS) |
| [AssembleNet: Searching for Multi-Stream Neural Connectivity in Video Architectures](https://arxiv.org/pdf/1905.13209v4.pdf) | ICLR | EA | - |
| How to Own the NAS in Your Spare Time | ICLR | - | - |
| [Fast, Accurate and Lightweight Super-Resolution with Neural Architecture Search](https://arxiv.org/pdf/1901.07261.pdf) | ICPR | G | [github](https://github.com/falsr/FALSR) |

## 2019 Venues

|  Title  |   Venue  |   Type   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [Self-Tuning Networks: Bilevel Optimization of Hyperparameters using Structured Best-Response Functions](https://arxiv.org/abs/1903.03088) | ICLR | - | - |
| [DATA: Differentiable ArchiTecture Approximation](http://papers.nips.cc/paper/8374-data-differentiable-architecture-approximation) | NeurIPS | - | - |
| [Random Search and Reproducibility for Neural Architecture Search](https://arxiv.org/pdf/1902.07638v3.pdf) | UAI | G | [GitHub](https://github.com/D-X-Y/NAS-Projects/blob/master/scripts-search/algos/RANDOM-NAS.sh) |
| [Improved Differentiable Architecture Search for Language Modeling and Named Entity Recognition](https://www.aclweb.org/anthology/D19-1367.pdf/) | EMNLP | G | - |
| [Continual and Multi-Task Architecture Search](https://www.aclweb.org/anthology/P19-1185.pdf) | ACL | RL | - |
| [Progressive Differentiable Architecture Search: Bridging the Depth Gap Between Search and Evaluation](https://arxiv.org/pdf/1904.12760v1.pdf) | ICCV | G | [GitHub](https://github.com/chenxin061/pdarts) |
| [Multinomial Distribution Learning for Effective Neural Architecture Search](https://arxiv.org/pdf/1905.07529v3.pdf) | ICCV | - | [GitHub](https://github.com/tanglang96/MDENAS) |
| [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244v5.pdf) | ICCV | EA | - |
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
| [DetNAS: Backbone Search for Object Detection](https://arxiv.org/pdf/1903.10979v4.pdf) | NeurIPS | EA | [GitHub](https://github.com/megvii-model/DetNAS) |
| [SpArSe: Sparse Architecture Search for CNNs on Resource-Constrained Microcontrollers](https://arxiv.org/pdf/1905.12107v1.pdf) | NeurIPS | - | - |
| [Efficient Forward Architecture Search](https://arxiv.org/abs/1905.13360) | NeurIPS | G | [Github](https://github.com/microsoft/petridishnn) |
| Efficient Neural ArchitectureTransformation Search in Channel-Level for Object Detection | NeurIPS | G | - |
| [XNAS: Neural Architecture Search with Expert Advice](https://arxiv.org/pdf/1906.08031v1.pdf) | NeurIPS | G | [GitHub](https://github.com/NivNayman/XNAS) |
| [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) | ICLR | G | [github](https://github.com/quark0/darts) |
| [ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://openreview.net/pdf?id=HylVB3AqYm) | ICLR | RL/G | [github](https://github.com/MIT-HAN-LAB/ProxylessNAS) |
| [Graph HyperNetworks for Neural Architecture Search](https://arxiv.org/pdf/1810.05749.pdf) | ICLR | G | - |
| [Learnable Embedding Space for Efficient Neural Architecture Compression](https://openreview.net/forum?id=S1xLN3C9YX) | ICLR | Other | [github](https://github.com/Friedrich1006/ESNAC) |
| [Efficient Multi-Objective Neural Architecture Search via Lamarckian Evolution](https://arxiv.org/abs/1804.09081) | ICLR | EA | - |
| [SNAS: stochastic neural architecture search](https://openreview.net/pdf?id=rylqooRqK7) | ICLR | G | - |
| [NetTailor: Tuning the Architecture, Not Just the Weights](https://arxiv.org/abs/1907.00274) | CVPR | G | [Github](https://github.com/pedro-morgado/nettailor) |
| [Searching for A Robust Neural Architecture in Four GPU Hours](http://xuanyidong.com/publication/gradient-based-diff-sampler/) | CVPR | G | [Github](https://github.com/D-X-Y/NAS-Projects) |
| [ChamNet: Towards Efficient Network Design through Platform-Aware Model Adaptation](http://openaccess.thecvf.com/content_CVPR_2019/papers/Dai_ChamNet_Towards_Efficient_Network_Design_Through_Platform-Aware_Model_Adaptation_CVPR_2019_paper.pdf) | CVPR | - | - |
| [Partial Order Pruning: for Best Speed/Accuracy Trade-off in Neural Architecture Search](https://arxiv.org/pdf/1903.03777.pdf) | CVPR | EA | [github](https://github.com/lixincn2015/Partial-Order-Pruning) |
| [FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search](https://arxiv.org/abs/1812.03443) | CVPR | G | - |
| [RENAS: Reinforced Evolutionary Neural Architecture Search](https://arxiv.org/abs/1808.00193) | CVPR | G | - |
| [Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation](https://arxiv.org/pdf/1901.02985.pdf) | CVPR |  G | [GitHub](https://github.com/tensorflow/models/tree/master/research/deeplab) |
| [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/abs/1807.11626) | CVPR | RL | [Github](https://github.com/AnjieZheng/MnasNet-PyTorch) |
| [MFAS: Multimodal Fusion Architecture Search](https://arxiv.org/pdf/1903.06496.pdf) | CVPR | EA | - |
| [A Neurobiological Evaluation Metric for Neural Network Model Search](https://arxiv.org/pdf/1805.10726.pdf) | CVPR | Other | - |
| [Fast Neural Architecture Search of Compact Semantic Segmentation Models via Auxiliary Cells](https://arxiv.org/abs/1810.10804) | CVPR | RL | - |
| [Customizable Architecture Search for Semantic Segmentation](https://arxiv.org/pdf/1908.09550v1.pdf) | CVPR | - | - |
| [Regularized Evolution for Image Classifier Architecture Search](https://arxiv.org/pdf/1802.01548.pdf) | AAAI | EA | - |
| AutoAugment: Learning Augmentation Policies from Data | CVPR | RL | - |
| Population Based Augmentation: Efficient Learning of Augmentation Policy Schedules | ICML | EA | - |
| [The Evolved Transformer](https://arxiv.org/pdf/1901.11117.pdf) | ICML | EA | [Github](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/evolved_transformer.py) |
| [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946v5.pdf) | ICML | RL | - |
| [NAS-Bench-101: Towards Reproducible Neural Architecture Search](https://arxiv.org/abs/1902.09635) | ICML | Other | [Github](https://github.com/google-research/nasbench) |
| [On Network Design Spaces for Visual Recognition](https://arxiv.org/abs/1905.13214) | ICCV | G | [Github](https://github.com/facebookresearch/nds) |

## 2018 Venues

|  Title  |   Venue  |   Type   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [Towards Automatically-Tuned Deep Neural Networks](https://link.springer.com/content/pdf/10.1007%2F978-3-030-05318-5.pdf) | BOOK | - | [GitHub](https://github.com/automl/Auto-PyTorch) |
| [NetAdapt: Platform-Aware Neural Network Adaptation for Mobile Applications](https://arxiv.org/pdf/1804.03230.pdf) | ECCV | - | [github](https://github.com/denru01/netadapt) |
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

## 2017 Venues

|  Title  |   Venue  |   Type   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578) | ICLR | RL | - |
| [Designing Neural Network Architectures using Reinforcement Learning](https://openreview.net/pdf?id=S1c2cvqee) | ICLR | RL | - | [github](https://github.com/bowenbaker/metaqnn) |
| [Neural Optimizer Search with Reinforcement Learning](http://proceedings.mlr.press/v70/bello17a/bello17a.pdf) | ICML | RL | - | [Large-Scale Evolution of Image Classifiers](https://arxiv.org/pdf/1703.01041.pdf) | ICML | EA | - |
| [Learning Curve Prediction with Bayesian Neural Networks](http://ml.informatik.uni-freiburg.de/papers/17-ICLR-LCNet.pdf) | ICLR | PD | - |
| [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://arxiv.org/abs/1603.06560) | ICLR | PD | - |
| [Hyperparameter Optimization: A Spectral Approach](https://arxiv.org/abs/1706.00764) | NeurIPS-W | Other | [github](https://github.com/callowbird/Harmonica) |
| Learning to Compose Domain-Specific Transformations for Data Augmentation | NeurIPS | - | - |

## Previous Venues

2012-2016

|  Title  |   Venue  |   Type   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [Speeding up Automatic Hyperparameter Optimization of Deep Neural Networksby Extrapolation of Learning Curves](http://ml.informatik.uni-freiburg.de/papers/15-IJCAI-Extrapolation_of_Learning_Curves.pdf) | IJCAI | PD | [github](https://github.com/automl/pylearningcurvepredictor) |

## arXiv

|  Title  |   Date  |   Type   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [NSGA-NET: A Multi-Objective Genetic Algorithm for Neural Architecture Search](https://arxiv.org/pdf/1810.03522.pdf) | 2018.10 | EA | - |
| [Training Frankenstein’s Creature to Stack: HyperTree Architecture Search](https://arxiv.org/pdf/1810.11714.pdf) | 2018.10 | G | - |
| [Population Based Training of Neural Networks](https://arxiv.org/abs/1711.09846) | 2017.11 | EA | [GitHub](https://github.com/MattKleinsmith/pbt) |

# Awesome Surveys

|  Title  |   Venue  |   Year   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [A Comprehensive Survey of Neural Architecture Search: Challenges and Solutions](https://arxiv.org/pdf/2006.02903.pdf) | ACM Computing Surveys | 2021 | - |
| [Automated Machine Learning on Graphs: A Survey](https://arxiv.org/pdf/2103.00742v3.pdf) | ICLR-W | 2021 | [GitHub](https://github.com/THUMNLab/AutoGL) |
| [On Hyperparameter Optimization of Machine Learning Algorithms: Theory and Practice](https://arxiv.org/pdf/2007.15745.pdf) | Neurocomputing | 2020 |[github](https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms) |
| [AutonoML: Towards an Integrated Framework for Autonomous Machine Learning](https://arxiv.org/pdf/2012.12600.pdf) | arXiv | 2020 | - |
| [Automated Machine Learning](https://link.springer.com/book/10.1007/978-3-030-05318-5) | Springer Book | 2019 | - |
| [Neural architecture search: A survey](http://www.jmlr.org/papers/volume20/18-598/18-598.pdf) | JMLR | 2019 | - |
| [AutoML: A Survey of the State-of-the-Art](https://arxiv.org/pdf/1908.00709.pdf) | arXiv | 2019 | [GitHub](https://github.com/marsggbo/automl_a_survey_of_state_of_the_art) |
| [A Survey on Neural Architecture Search](https://arxiv.org/pdf/1905.01392.pdf) | arXiv | 2019 | - |
| [Taking human out of learning applications: A survey on automated machine learning](https://arxiv.org/pdf/1810.13306.pdf) | arXiv | 2018 | - |
