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

<center class="half">
  <img src="/images/lifelong/Lifelong.png" width="760" height="200" />
</center>

Machine learning algorithms can now solve many problems even better than humans. However, machines are still far from being intelligent that they typically need to relearn when facing new tasks, while humans are able to learn new things efficiently by ultilizing learned knowledge. This motivates the study called lifelong learning (Thrun and Pratt, 1998), which aims to perform better over time by **transferring** information learned from previously tasks to later ones, *under the belief that there are some commonalities across tasks*. 

To formulate the commonalities across tasks, we assume that 
>Tasks are related as they share some common representation, but they are different as each requires a different predictor on top of the representation to form a classifier.

In this paper, we consider the following learning problem: 

# Notations and Settings

* representation space : $\mathcal{G}$ (usually very large)
* predictor space : $\mathcal{H}$
* number of tasks : $K$
* number of samples in each task : $T_k, \forall k\in [K]$ and $\sum T_k =T$
* loss functions are $[0,1]$-valued

Both the tasks and their samples arrive sequentially so that in each time step, we only see a sample from one single task. For each task $k$ and each step $s$ in it, we need to choose a representation $g_{k,s}$ and an accompanying predictor $h_{k,s}$  which jointly provide a decision(classifier) for us. After making this decision, we suffer some loss $\ell_{k,s}(g_{k,s}, h_{k,s})$ according to loss function $\ell_{k,s}$, receive some feedback information, and then proceed to the next iteration.

To measure the performance of a learning algorithm, different settings have their own natural choices. Since the samples of each task arrive one after one, an often adopted measure is the **regret**. Furthermore, to capture the assumption above, we measure the regret by comparing against an offline algorithm which must use a fixed representation for all the tasks but is allowed to use different predictors for different tasks.

\[ 
Regret(T)=\sum\limits_{k,s}\ell_{k,s}(g_{k,s}, h_{k,s})-\min_{g, h_1, \cdots, h_K}\sum\limits_{k,s}\ell_{k,s}(g, h_k) 
\]

## Example


Before going to details, here we give a simple example to show possible advantages of lifelong learning over relearning tasks.
We start from the case that $|\mathcal{G}|$ and $|\mathcal{H}|$ are finite but the loss functions are arbitrary. 
For this, we provide an efficient algorithm achieving a regret of 
$\mathcal{O}\left(\sqrt{T\log |\mathcal{G}|}+\sqrt{TK\log |\mathcal{H}|}\right)$,
while relearning the representation results in regret of


\[\sum\limits_{k=1}^{K}\mathcal{O}\left(\sqrt{T_{k}\log \mathcal{G}}+\sqrt{T_{k}\log \mathcal{H}}\right)\leq\mathcal{O}\left(\sqrt{KT\log \mathcal{G}}+\sqrt{KT\log \mathcal{H}}\right)\]


First of all, our bound prevents the number of tasks from affecting the learning of representations. That is to say, the regret of learning the representations doesn't grow with the number of tasks (for a fixed $T$). Since $\mathcal{G}$ is usually large, this benefit makes our bound attractive for large $K$.

Moreover, as learning the representations is typically much more costly than learning predictors in lifelong learning, if under some conditions it is then possible to identify the best representation $g^{*}$ for all tasks at some step $t<T$, this would allow us to learn new tasks faster, by saving the time for learning the representation.

# Full-Information Adversarial Setting

In learning problems, we always guide the learning by losses. However, here the losses $\ell_{k,s}(g_{k,s}, h_{k,s})$ depend on both the representation and the predictor. This makes learning harder. 

If we already know what the best representation $g^{*}$ is, it remains to learn predictors for each task. However, how can we estimate how good a representation is when **a good representation may look bad if we choose a bad predictor to go with it**? A sensible choice seems to be accompanying it with its *best predictor in a task *. That is, to measure a representation $g$ in task $k$ by $\hat{L}_k (g)$, where

$$ 
\hat{L}_k(g)=\min_h \sum\limits_{s=1}^{T_k} \ell_{k,s}(h, g). 
$$

The best representation throughout tasks is then nature to be 

$$
\min_{g}\sum\limits_{k=1}^K \hat{L}_k (g).
$$

Everything goes well so far. Nevertheless, the above method only provides us with measurement at the end of tasks. When learning within a task, we do not know what the best predictor of a representation is as **the predictor which looks best so far may turn out to be bad at the end of the task in the adversarial setting**. This is perhaps one reason why Alquier et al. (2017) chose to update their representations only at the end of each task, where the best predictor of each representation in the task is ensured. This consequently requires a large number of tasks in order to have a good regret bound due to less update.

To achieve our regret bound, we have to construct appropriate loss functions so as to update representations at every step.


recall that we hope to learn the representations continuously through time using all the data across different tasks, while we still have to relearn predictors for different tasks. To do that, we would like to decouple the learning of representations from that of predictors, for them to have different learning algorithms as well as different learning schedules.



There were some empirical studies on the possibility of evolving the network structures over different tasks to do lifelong learning (Rusu et al., 2016; Lee et al., 2017). For theoretical studies, we, and most prior works as well, focused on learning with fixed architectures. That is, the representation and the predictor spaces are fixed before receiving samples. 



