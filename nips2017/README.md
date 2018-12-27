## Format of summary

1.  keyword
2.  the question studied
3.  why it is worth studying that question
4.  the methods used
5.  experiments
6.  the basic results



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
