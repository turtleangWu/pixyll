---
layout: post
title: Lifelong Learning with Low Regret
date: 2019-03-30 12:00
summary: Lifelong learning comes from online learning and multi-task learning. We face tasks and samples sequence by sequence as usual online learning settings. However, there are more than one task which makes learning more difficult.
categories: lifelong
author: Yi-Shan Wu
visible: True
---

# Introduction

Machine learning algorithms can now solve many problems even better than humans. However, machines are still far from being intelligent that they typically need to relearn when facing new tasks, while humans are able to learn new things efficiently by ultilizing learned knowledge. This motivates the study called lifelong learning (Thrun and Pratt, 1998), which aims to perform better over time by transferring information learned from previously tasks to later ones, under the belief that there are some commonalities across tasks. 

There were some empirical studies on the possibility of evolving the network structures over different tasks to do lifelong learning (Rusu et al., 2016; Lee et al., 2017). For theoretical studies, we, and most prior works as well, focused on learning with fixed architectures. 

<center class="half">
  <img src="/images/lifelong/Lifelong.png" width="760" height="200" />
</center>

In this paper, we consider the following learning problem: 

# Notations and Settings

* representation space : $\mathcal{G}$ (usually very large)
* predictor space : $\mathcal{H}$
* number of tasks : $K$
* number of samples in each task : $T_k, \forall k\in [K]$ and $\sum T_k =T$

Both the tasks and their samples arrive sequentially so that in each time step, we only see a sample from one single task (Alquier et al., 2017). For each task $k$ and each step $s$ in it, we need to choose a representation $g_{k,s}$ and an accompanying predictor $h_{k,s}$  which jointly provide a decision for us. After making this decision, we suffer some loss $\ell_{k,s}(g_{k,s}, h_{k,s})$ according to some loss function $\ell_{k,s}$, receive some feedback information, and then proceed to the next iteration.

As introduced above, tasks are related as they share some common representation, but they are different as each requires a different predictor on top of the representation. As learning the representations is typically much more costly than learning predictors in lifelong learning, we would like to understand if it is possible to learn them continuously through time across different tasks, instead of relearning.

To measure the performance of a learning algorithm, different settings have their own natural choices. Since the samples of each task arrive one after one, an often adopted measure is the regret.

\[ \sum\limits_{k,s}\ell_{k,s}(g_{k,s}, h_{k,s})-\min_{g, h_1, \cdots, h_K}\sum\limits_{k,s}\ell_{k,s}(g, h_k) \]

# Example


Before dive into details, here we give a simple example to show possible advantages of lifelong learning over relearning tasks.
We start from the case that $|\mathcal{G}|$ and $|\mathcal{H}|$ are finite but the loss functions are arbitrary. 
For this, we provide an efficient algorithm achieving a regret of 
$\mathcal{O}\left(\sqrt{T\log |\mathcal{G}|}+\sqrt{TK\log |\mathcal{H}|}\right)$,
while relearning the representation for each task has a regret of 


\[\sum\limits_{k=1}^K\mathcal{O}\left(\sqrt{T_k\log |\mathcal{G}|}+\sqrt{T_k\log |\mathcal{H}|}\right)\leq\mathcal{O}\left(\sqrt{KT\log |\mathcal{G}|}+\sqrt{TK\log |\mathcal{H}|}\right)$.\]

As mentioned above, learning representations is usually more costly than learning predictors. Our bound prevents the number of tasks from affecting the learning of representations. That is to say, the cost of learning the representations doesn't grow with the number of tasks.



