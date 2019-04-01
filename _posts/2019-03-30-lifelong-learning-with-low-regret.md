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




