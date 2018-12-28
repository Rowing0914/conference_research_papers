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
  2. Prerequisite: **Subspace clustering** is an extension of traditional **cluster**- ing that seeks to find **clusters** in different **subspaces** within a dataset. Often in high dimensional data, many dimen- sions are irrelevant and can mask existing **clusters** in noisy data by Lance Parsons et al., 2004(https://www.kdd.org/exploration_files/parsons.pdf)
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
  1. keyword
  2. Prerequisites
  3. the question studied
  4. why it is worth studying that question
  5. the methods used
  6. experiments
  7. the basic results
