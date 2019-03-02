## Format of summary

1.  keyword
2.  Prerequisites
3.  the question studied
4.  why it is worth studying that question
5.  the methods used
6.  experiments
7.  the basic results



- [Wider and Deeper, Cheaper and Faster: Tensorized LSTMs for Sequence Learning](https://papers.nips.cc/paper/6606-wider-and-deeper-cheaper-and-faster-tensorized-lstms-for-sequence-learning) [Zhen He](https://papers.nips.cc/author/zhen-he-9357), [Shaobing Gao](https://papers.nips.cc/author/shaobing-gao-9358), [Liang Xiao](https://papers.nips.cc/author/liang-xiao-9359), [Daxue Liu](https://papers.nips.cc/author/daxue-liu-10481), [Hangen He](https://papers.nips.cc/author/hangen-he-10482), [David Barber](https://papers.nips.cc/author/david-barber-1451)
  1. keyword: LSTM, computational efficiency, wider and deeper, tensors, cross-layer convolution
  2. the question studied: how to widen and deepen LSTM efficiently without introducing unnecessary complexity in the nets.
  3. why it is worth studying that question: normally the width and depth of the network correlate to the complexity of the network, and that causes the inefficiency in training.
  4. the methods used: Tensorised LSTM in which the hidden states are represented by tensors and updated via a cross-layer convolution.
  5. experiments: Experiments conducted on five challenging sequence learning tasks show the potential of the proposed model
  6. the basic results: by increasing the tensor size the network can be deepened implicitly without additional parameters to tune.
- [Concentration of Multilinear Functions of the Ising Model with Applications to Network Data](https://papers.nips.cc/paper/6607-concentration-of-multilinear-functions-of-the-ising-model-with-applications-to-network-data) [Constantinos Daskalakis](https://papers.nips.cc/author/constantinos-daskalakis-8166), [Nishanth Dikkala](https://papers.nips.cc/author/nishanth-dikkala-9360), [Gautam Kamath](https://papers.nips.cc/author/gautam-kamath-8167)
  1. keyword, Ising model, near-right concentration, social networks
  2. Prerequisite: The Ising model, named after the physicist Ernst Ising, is a mathematical model of ferromagnetism in statistical mechanics. The model consists of discrete variables that represent magnetic dipole moments of atomic spins that can be in one of two states. [Wikipedia](https://en.wikipedia.org/wiki/Ising_model)
  3. the question studied: r main technical contribution is to obtain near-tight concentration inequalities for polynomial functions of the Ising model, whose concentration radii are tight up to logarithmic factors.
  4. why it is worth studying that question
  5. the methods used
  6. experiments: demonstrate the efficacy of such functions as statistics for testing the strength of interactions in social networks in both synthetic and real world data.
  7. the basic results: the result indicates that the optimality of their approach up to logarithmic factors in the dimension
- [Deep Subspace Clustering Networks](https://papers.nips.cc/paper/6608-deep-subspace-clustering-networks) [Pan Ji](https://papers.nips.cc/author/pan-ji-9361), [Tong Zhang](https://papers.nips.cc/author/tong-zhang-1901), [Hongdong Li](https://papers.nips.cc/author/hongdong-li-9362), [Mathieu Salzmann](https://papers.nips.cc/author/mathieu-salzmann-4557), [Ian Reid](https://papers.nips.cc/author/ian-reid-8117)
  1. keyword: subspace clustering, unsupervised subspace, neural network, deep auto-encoders, self-expressiveness
  2. Prerequisite: **Subspace clustering** is an extension of traditional **clustering** that seeks to find **clusters** in different **subspaces** within a dataset. Often in high dimensional data, many dimensions are irrelevant and can mask existing **clusters** in noisy data by Lance Parsons et al., 2004(https://www.kdd.org/exploration_files/parsons.pdf)
  3. the question studied: this paper tackles the problem of subspace clustering performance
  4. why it is worth studying that question: new self-expressive layer can provide a simple but effective way to learn pairwise affinities between all data points through a standard back-propagation procedure based on being differentiable and it can deal with the non-linearity in the data structure.
  5. the methods used: introduce a novel self-expressive layer between the encoder and decoder to mimic the "self-expressiveness" property being known for the efficiency in the traditional subspace clustering literature.
  6. experiments: implement the algo in python with Tensorflow and examined it with four standard
     datasets, i.e., the Extended Yale B and ORL face image datasets, and the COIL20/100 object image
     datasets.
  7. the basic results: their experiments show that the proposed method significantly outperforms the state-of-the-art unsupervised subspace clustering methods.
- [Attentional Pooling for Action Recognition](https://papers.nips.cc/paper/6609-attentional-pooling-for-action-recognition) [Rohit Girdhar](https://papers.nips.cc/author/rohit-girdhar-9363), [Deva Ramanan](https://papers.nips.cc/author/deva-ramanan-2563)
  1. keyword: attention mechanism, action recognition, low-rank approximations, bilinear pooling methods, fine-grained representations
  2. Prerequisites: Attention mechanisms(Attention Mechanisms in Neural Networks are (very) loosely based on the visual attention mechanism found in humans. Human visual attention is well-studied and while there exist different models, all of them essentially come down to being able to focus on a certain region of an image with “high resolution” while perceiving the surrounding image in “low resolution”, and then adjusting the focal point over time from [WildML](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/).)
  3. the question studied: how to incorporate attention mechanisms in action recognition and human object interaction tasks
  4. why it is worth studying that question
  5. the methods used: integration of bottom-up saliency and top-down attention
  6. experiments: experiment with three recent, large scale action recognition datasets, across still images and videos, namely MPII, HICO and HMDB51
  7. the basic results: model produces competitive or state-of-the-art results on widely benchmarked datasets, by learning where to look when pooling features across an image. About 12.5% relative improvement.
- [On the Consistency of Quick Shift](https://papers.nips.cc/paper/6610-on-the-consistency-of-quick-shift) [Heinrich Jiang](https://papers.nips.cc/author/heinrich-jiang-9364)
  1. keyword: Quick Shift, consistency
  2. Prerequisites: [Quick Shift](http://vision.cs.ucla.edu/papers/vedaldiS08quick.pdf) It is simple and proceeds as follows: it moves each sample to its closest sample with a higher empirical density if one exists in a τ radius ball, where the empirical density is taken to be the Kernel Density Estimator (KDE). The output of the procedure can thus be seen as a graph whose vertices are the sample points and a directed edge from each sample to its next point if one exists.
  3. the question studied: consistency guarantees for Quick Shift under mild assumptions
  4. why it is worth studying that question
  5. the methods used: they showed that Quick Shift recovers the modes of a density from a finite sample with minimax optimal guarantees.
  6. experiments
  7. the basic results: they demonstrated a procedure for modal regression using Quick Shift which attains strong statistical guarantees.
- [Breaking the Nonsmooth Barrier: A Scalable Parallel Method for Composite Optimization](https://papers.nips.cc/paper/6611-breaking-the-nonsmooth-barrier-a-scalable-parallel-method-for-composite-optimization) [Fabian Pedregosa](https://papers.nips.cc/author/fabian-pedregosa-9365), [Rémi Leblond](https://papers.nips.cc/author/remi-leblond-9366), [Simon Lacoste-Julien](https://papers.nips.cc/author/simon-lacoste-julien-7161)
  1. keyword: parallel asynchronous variants of stochastic gradient descent, multi-core architecture
  2. the question studied: they addressed the non-applicability of multi-core architecture for ML algo of non-smooth algorithms, like LASSO.
  3. the methods used: PROXASAGA, advanced version of SAGA
  4. experiments: they focused on $l_1 + l_2$ regularised logistic regression
  5. the basic results: they proved that PROXASAGA can speed-up learning up to 12x on a 20-core machine.
- [Dual-Agent GANs for Photorealistic and Identity Preserving Profile Face Synthesis](https://papers.nips.cc/paper/6612-dual-agent-gans-for-photorealistic-and-identity-preserving-profile-face-synthesis) [Jian Zhao](https://papers.nips.cc/author/jian-zhao-9367), [Lin Xiong](https://papers.nips.cc/author/lin-xiong-9368), [Panasonic Karlekar Jayashree](https://papers.nips.cc/author/panasonic-karlekar-jayashree-9369), [Jianshu Li](https://papers.nips.cc/author/jianshu-li-9370), [Fang Zhao](https://papers.nips.cc/author/fang-zhao-6453), [Zhecan Wang](https://papers.nips.cc/author/zhecan-wang-9371), [Panasonic Sugiri Pranata](https://papers.nips.cc/author/panasonic-sugiri-pranata-9372), [Panasonic Shengmei Shen](https://papers.nips.cc/author/panasonic-shengmei-shen-9373), [Shuicheng Yan](https://papers.nips.cc/author/shuicheng-yan-4679), [Jiashi Feng](https://papers.nips.cc/author/jiashi-feng-6686)
  1. keyword: Synthesising realistic profile faces, face recognition
  2. the question studied: how to accurately detect the human faces from the people's profiles images in which people take some weird positioning and the faces are not clearly captured
  3. Proposition: DA-GAN(dual-agent generative adversarial network)
     1. a pose perception loss
     2. an identity perception loss
     3. an adversarial loss with a boundary equilibrium regularisation term
  4. experiments: NIST IJB-A unconstrained face recognition benchmark
  5. the basic results: **DA-GAN** not only presents compelling perceptual results but also significantly
     outperforms state-of-the-arts on the large-scale and challenging NIST IJB-A unconstrained face recognition benchmark.
- [Dilated Recurrent Neural Networks](https://papers.nips.cc/paper/6613-dilated-recurrent-neural-networks) [Shiyu Chang](https://papers.nips.cc/author/shiyu-chang-9374), [Yang Zhang](https://papers.nips.cc/author/yang-zhang-9375), [Wei Han](https://papers.nips.cc/author/wei-han-9376), [Mo Yu](https://papers.nips.cc/author/mo-yu-7675), [Xiaoxiao Guo](https://papers.nips.cc/author/xiaoxiao-guo-6764), [Wei Tan](https://papers.nips.cc/author/wei-tan-9377), [Xiaodong Cui](https://papers.nips.cc/author/xiaodong-cui-9378), [Michael Witbrock](https://papers.nips.cc/author/michael-witbrock-9379), [Mark A. Hasegawa-Johnson](https://papers.nips.cc/author/mark-a-hasegawa-johnson-9380), [Thomas S. Huang](https://papers.nips.cc/author/thomas-s-huang-2118)
  1. keyword: RNN, **dilated recurrent skip connections**
  2. the question studied: Learning with recurrent neural networks (RNNs) on long sequences is a notoriously difficult task.
     1. complex dependencies
     2. vanishing and exploding gradients
     3. efficient parallelisation
  3. Proposition: In this paper, we introduce a simple yet effective RNN connection structure, the DILATEDRNN, which simultaneously tackles all challenges above. 
  4. experiments: they examined the proposition on four tasks
     1. long-term memorisation
     2. pixel-by-pixel MNIST classification
     3. character-level language modelling on the Penn Treebank
     4. speaker identification with raw waveforms on VCTK
  5. the basic results: the DILATEDRNN reduces the number of parameters needed and enhances training efficiency significantly, while matching state-of-the-art performance (even with standard RNN cells) in tasks involving very long-term dependencies.
- [Hunt For The Unique, Stable, Sparse And Fast Feature Learning On Graphs](https://papers.nips.cc/paper/6614-hunt-for-the-unique-stable-sparse-and-fast-feature-learning-on-graphs) [Saurabh Verma](https://papers.nips.cc/author/saurabh-verma-9381), [Zhi-Li Zhang](https://papers.nips.cc/author/zhi-li-zhang-9382)
  1. keyword: learning on grpahs
  2. the question studied: 
  3. Proposition: FGSD(family of graph spectral distances) for graph classification problem
  4. experiments: 
  5. the basic results: it significantly outperforms all the more sophisticated state-of-art algorithms on the unlabeled node datasets in terms of both accuracy and speed; it also yields very competitive results on the labeled datasets – despite the fact it does not utilize any node label information.
- [Scalable Generalized Linear Bandits: Online Computation and Hashing](https://papers.nips.cc/paper/6615-scalable-generalized-linear-bandits-online-computation-and-hashing) [Kwang-Sung Jun](https://papers.nips.cc/author/kwang-sung-jun-8224), [Aniruddha Bhargava](https://papers.nips.cc/author/aniruddha-bhargava-5025), [Robert Nowak](https://papers.nips.cc/author/robert-nowak-2383), [Rebecca Willett](https://papers.nips.cc/author/rebecca-willett-3138)
  1. keyword: Generalized Linear Bandits (GLBs), a natural extension of the stochastic linear
     bandits
  2. the question studied: existing GLBs scale poorly with the number of rounds and the number of arms, limiting their utility in practice. 
  3. Proposition: **GLOC**(Generalized Linear extension of the Online-to-confidence-set
     Conversion).  This paper proposes new, scalable solutions to the GLB problem in two respects. unlike existing GLBs, whose per-time-step space and time complexity grow at least linearly with time, they propose a new algorithm that performs online computations to enjoy a constant space and time complexity
  4. experiments: 
  5. the basic results: their proposition outperforms others in terms of the complexity of algorithm
- [Probabilistic Models for Integration Error in the Assessment of Functional Cardiac Models](https://papers.nips.cc/paper/6616-probabilistic-models-for-integration-error-in-the-assessment-of-functional-cardiac-models) [Chris Oates](https://papers.nips.cc/author/chris-oates-8054), [Steven Niederer](https://papers.nips.cc/author/steven-niederer-9383), [Angela Lee](https://papers.nips.cc/author/angela-lee-9384), [François-Xavier Briol](https://papers.nips.cc/author/francois-xavier-briol-8053), [Mark Girolami](https://papers.nips.cc/author/mark-girolami-2744)
  1. keyword: functional cardiac models, approximating the integrals by a prediction model
  2. the question studied: For the functional cardiac models that motivate this work, neither f nor p possess a closed-form expression and evaluation of either requires $\approx$ 100 CPU hours, precluding standard numerical integration methods. they wanted to reduce this time by approximating the integrals outputs by probabilistic models
  3. Proposition: Our proposal is to treat integration as an estimation problem, with a joint model for both the a priori unknown function f and the a priori unknown distribution p
  4. experiments: 
  5. the basic results: The result is a posterior distribution over the integral that explicitly accounts for dual sources of numerical approximation error due to a severely limited computational budget.
- [Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent](https://papers.nips.cc/paper/6617-machine-learning-with-adversaries-byzantine-tolerant-gradient-descent) [Peva Blanchard](https://papers.nips.cc/author/peva-blanchard-9385), [El Mahdi El Mhamdi](https://papers.nips.cc/author/el-mahdi-el-mhamdi-9386), [Rachid Guerraoui](https://papers.nips.cc/author/rachid-guerraoui-9387), [Julien Stainer](https://papers.nips.cc/author/julien-stainer-9388)
  1. keyword: Byzantine failures, SGD
  2. the question studied: We study the resilience to Byzantine failures of distributed implementations of Stochastic Gradient Descent (SGD). So far, distributed machine learning frameworks have largely ignored the possibility of failures, especially arbitrary (i.e., Byzantine) ones. Causes of failures include software bugs, network asynchrony, biases in local datasets, as well as attackers trying to compromise the entire system.
  3. Proposition: Assuming a set of n workers, up to f being Byzantine, we ask how resilient can SGD be, without limiting the dimension, nor the size of the parameter space. **Krum**, an aggregation rule that satisfies our resilience property, which we argue is the first provably Byzantine-resilient algorithm for distributed SGD
  4. experiments: 
  5. the basic results: 
- [Dynamic Safe Interruptibility for Decentralized Multi-Agent Reinforcement Learning](https://papers.nips.cc/paper/6618-dynamic-safe-interruptibility-for-decentralized-multi-agent-reinforcement-learning) [El Mahdi El Mhamdi](https://papers.nips.cc/author/el-mahdi-el-mhamdi-9386), [Rachid Guerraoui](https://papers.nips.cc/author/rachid-guerraoui-9387), [Hadrien Hendrikx](https://papers.nips.cc/author/hadrien-hendrikx-9389), [Alexandre Maurer](https://papers.nips.cc/author/alexandre-maurer-9390)
  1. keyword: MARL
  2. the question studied: safety in RL is important though, previously defined concept of safe interruptibility is not simply extendable in MARL case.
  3. Proposition: This paper introduces dynamic safe interruptibility, an alternative definition more suited to decentralized learning problems, and studies this notion in two learning frameworks: joint action learners and independent learners
  4. experiments: 
  5. the basic results: We show however that if agents can detect interruptions, it is possible to prune the observations to ensure dynamic safe interruptibility even for independent learners.
- [Interactive Submodular Bandit](https://papers.nips.cc/paper/6619-interactive-submodular-bandit) [Lin Chen](https://papers.nips.cc/author/lin-chen-8726), [Andreas Krause](https://papers.nips.cc/author/andreas-krause-3758), [Amin Karbasi](https://papers.nips.cc/author/amin-karbasi-6295)
  1. keyword: submodular fuction
  2. the question studied: In many machine learning applications, submodular functions have been used as a model for evaluating the utility or payoff of a set, yet in many real life situations, however, the utility function is not fully known in advance and can only be estimated via interactions.
  3. Proposition: **SM-UCB**  We model such problems as an interactive submodular bandit optimization, where in each round we receive a context (e.g., previously selected movies) and have to choose an action (e.g., propose a new movie). We then receive a noisy feedback about the utility of the action (e.g., ratings) which we model as a submodular function over the context-action space.
  4. experiments: we evaluate our results on four concrete applications, including movie recommendation (on the MovieLense data set), news recommendation (on Yahoo! Webscope dataset), interactive influence maximization (on a subset of the Facebook network), and personalized data summarization (on Reuters Corpus).
  5. the basic results: In all these applications, we observe that SM-UCB consistently outperforms the prior art.
- [Learning to See Physics via Visual De-animation](https://papers.nips.cc/paper/6620-learning-to-see-physics-via-visual-de-animation) [Jiajun Wu](https://papers.nips.cc/author/jiajun-wu-8096), [Erika Lu](https://papers.nips.cc/author/erika-lu-9391), [Pushmeet Kohli](https://papers.nips.cc/author/pushmeet-kohli-4400), [Bill Freeman](https://papers.nips.cc/author/bill-freeman-4274), [Josh Tenenbaum](https://papers.nips.cc/author/josh-tenenbaum-6308)
  1. keyword: scene description languages, 3D reconstruction methods, simulation engines and virtual environments.
  2. the question studied: We introduce a paradigm for understanding physical scenes without human annotations. n forward simulation, inverting a physics or graphics engine is a computationally hard problem; we overcome this challenge by using a convolutional inversion network
  3. Proposition: We introduce a paradigm for understanding physical scenes without human annotations. At the core of our system is a physical world representation that is first recovered by a perception module and then utilized by physics and graphics engines
  4. experiments: We evaluate our system on both synthetic and real datasets involving multiple physical scenes
  5. the basic results: 
- [Label Efficient Learning of Transferable Representations acrosss Domains and Tasks](https://papers.nips.cc/paper/6621-label-efficient-learning-of-transferable-representations-acrosss-domains-and-tasks) [Zelun Luo](https://papers.nips.cc/author/zelun-luo-9392), [Yuliang Zou](https://papers.nips.cc/author/yuliang-zou-9393), [Judy Hoffman](https://papers.nips.cc/author/judy-hoffman-7393), [Li F. Fei-Fei](https://papers.nips.cc/author/li-f-fei-fei-7198)
  1. keyword: domain shift
  2. the question studied: domain shift problem in Transfer learning
  3. Proposition: We propose a framework that learns a representation transferable across different domains and tasks in a label efficient manner. Our approach battles domain shift with a domain adversarial loss, and generalises the embedding to novel task using a metric learning-based approach.
  4. experiments: 
     1. MNIST handwritten digits database
     2. Google Street View House Numbers (SVHN) 
  5. the basic results: Our method shows compelling results on novel classes within a new domain even when only a few labelled examples per class are available, outperforming the prevalent fine-tuning approach. In addition, we demonstrate the effectiveness of our framework on the transfer learning task from image object recognition to video action recognition.
- [Decoding with Value Networks for Neural Machine Translation](https://papers.nips.cc/paper/6622-decoding-with-value-networks-for-neural-machine-translation) [Di He](https://papers.nips.cc/author/di-he-9115), [Hanqing Lu](https://papers.nips.cc/author/hanqing-lu-9394), [Yingce Xia](https://papers.nips.cc/author/yingce-xia-9116), [Tao Qin](https://papers.nips.cc/author/tao-qin-3844), [Liwei Wang](https://papers.nips.cc/author/liwei-wang-4410), [Tie-Yan Liu](https://papers.nips.cc/author/tie-yan-liu-8972)
  1. keyword: neural machine translation, beam search
  2. the question studied: the problem in beam search is that since it only searches for local optima at each time step through one-step forward looking, it usually cannot output the best target sentence.
  3. Proposition: Inspired by the success and methodology of AlphaGo, in this paper we propose using a prediction network to improve beam search, which takes the source sentence x, the currently available decoding output y1, · · · , yt−1 and a candidate word w at step t as inputs and predicts the long-term value (e.g., BLEU score) of the partial target sentence if it is completed by the NMT model.
  4. experiments: 
  5. the basic results: Experiments show that such an approach can significantly improve the translation accuracy on several translation tasks.
- [Parametric Simplex Method for Sparse Learning](https://papers.nips.cc/paper/6623-parametric-simplex-method-for-sparse-learning) [Haotian Pang](https://papers.nips.cc/author/haotian-pang-9395), [Han Liu](https://papers.nips.cc/author/han-liu-3555), [Robert J. Vanderbei](https://papers.nips.cc/author/robert-j-vanderbei-9396), [Tuo Zhao](https://papers.nips.cc/author/tuo-zhao-5638)
  1. keyword: data analysis, high dimensional sparse data
  2. the question studied: High dimensional sparse learning has imposed a great computational challenge to large scale data analysis
  3. Proposition: 
  4. experiments: 
  5. the basic results: 
