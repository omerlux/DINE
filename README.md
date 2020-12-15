# DINE

Based on: [H. H. Permuter, Z. Goldfeld, D. Tsur and Z. Aharoni. Capacity of Continuous Channels with Memory via Directed Information Neural Estimator. arXiv preprint arXiv:2003.04179. May 2020.](https://arxiv.org/pdf/2003.04179v2.pdf)

> Abstract—Calculating the capacity (with or without feedback)
of channels with memory and continuous alphabets is a challenging task. It requires optimizing the directed information (DI) rate
over all channel input distributions. The objective is a multiletter expression, whose analytic solution is only known for a
few specific cases. When no analytic solution is present or the
channel model is unknown, there is no unified framework for
calculating or even approximating capacity. This work proposes
a novel capacity estimation algorithm that treats the channel
as a ‘black-box’, both when feedback is or is not present. The
algorithm has two main ingredients: (i) a neural distribution
transformer (NDT) model that shapes a noise variable into the
channel input distribution, which we are able to sample, and (ii)
the DI neural estimator (DINE) that estimates the communication
rate of the current NDT model. These models are trained by an
alternating maximization procedure to both estimate the channel
capacity and obtain an NDT for the optimal input distribution.
The method is demonstrated on the moving average additive
Gaussian noise channel, where it is shown that both the capacity
and feedback capacity are estimated without knowledge of the
channel transition kernel. The proposed estimation framework
opens the door to a myriad of capacity approximation results for
continuous alphabet channels that were inaccessible until now.

In this repo I reproduce the results of the above paper. To know more please take a look at [Dine Implementation Report](https://github.com/omerlux/DINE/blob/master/DINE%20Implementation.pdf).
