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

Moreover, as learning the representations is typically much more costly than learning predictors in lifelong learning, if under some conditions it is possible to identify the best representation $g^{\*}$ for all tasks at some step $t<T$, this would allow us to learn new tasks faster by saving the time for learning the representation.

# First Challenge

In learning problems, we always guide the learning by losses. However, here the losses $\ell_{k,s}(g_{k,s}, h_{k,s})$ depend on both the representation and the predictor. This makes learning harder. If we already know what the best representation $g^{\*}$ is, it remains to learn predictors for each task. However, how can we estimate how good a representation is when **a good representation may look bad if we choose a bad predictor to go with it**? A sensible choice seems to be accompanying it with its **best predictor in a task**. Take the full-information adversarial setting for example.


## Full-Information Adversarial Setting

In full-information setting, the whole loss fuction at each step is revealed. A sensible choice to measure a representation $g$ in task $k$ is $\hat{L}_k (g)$, where

$$ 
\hat{L}_k(g)=\min_h \sum\limits_{s=1}^{T_k} \ell_{k,s}(h, g). 
$$

The best representation throughout tasks is then nature to be 

$$
\min_{g}\sum\limits_{k=1}^K \hat{L}_k (g).
$$

Everything goes well so far. Nevertheless, the above method only provides us with measurement at the end of tasks. When learning within a task, we do not know what the best predictor of a representation is as **the predictor which looks best so far may turn out to be bad at the end of the task in the adversarial setting**. This is perhaps one reason why Alquier et al. (2017) chose to update their representations only at the end of each task, where the best predictor of each representation in the task is ensured. This consequently requires a large number of tasks in order to have a good regret bound due to less update. 

To achieve our regret bound, we have to construct appropriate loss functions so as to update representations more often (we actually update them at every step). 

### Idea for Full-Information Adversarial Setting

Recall that tasks are related as they share some common representation, but they are different as each requires a different predictor on top of the representation. Therefore, we hope to **learn the representations continuously through time**, while we still have to **relearn predictors for different tasks**. To do that, we would like to decouple the learning of representations from that of predictors, for them to have different loss functions and different learning schedules. For finite representations, we describe our solution via a generic algorithm.

### Algorithm

We take $alg_G$ to learn the representation and have it update continuously through time across different tasks. For each possible representation $g$, we have a separate copy of $alg^{(g)}_{H}$ for learning the accompanying predictors. When starting a new task $k$, reset each copy $alg^{(g)}_H$ and redo its learning.


At step $s$ in task $k$, we sample a representation $g_{k,s}$ according to the distribution $G_{k,s}$ of $alg_{G}$, followed by sampling a predictor $h_{k,s}$ according to the distribution $H_{k,s}^{(g_{k,s})}$ of $alg_{H}^{(g_{k,s})}$. The joint action we play is $(g_{k,s}, h_{k,s})$ and the loss function is $\ell_{k,s}$. Then we update the distribution of $alg_{G}$ using the loss function on $g$ defined as

$$\hat{\ell}_{k,s}(g)=\mathbb{E}_{h\sim H_{k,s}^{(g)}}\left[\ell_{k,s}(g,h) \right] $$

and update the distribution of each copy $alg_{H}^{(g)}$ using $\ell_{k,s}\left( g,\cdot \right)$ as the loss function on predictors. That is, the loss of $g$ at each step is defined to be the average loss of $g$ with its predictors. With this algorithm, we have the following theorem.

>**Theorem :**
>Suppose the $t$-step regret bounds of $alg_G$ and $alg_H$ are $reg_{G}(t)$ and $reg_{H}(t)$, respectively. Then the $T$-step regret bound of our algorithm is at most $reg_G\left(T\right) +\sum\limits_{k=1}^{K} reg_H\left(T_k \right)$.

Now for adversarial cases, if we use multiplicative update (MU) algorithm as $alg_G$ and $alg_H$, there is a corollary, which is also our result for above example:

>**Corollary :**
>For the case that $|\mathcal{G}|$ and $|\mathcal{H}|$ are finite but the loss functions are arbitrary, we provide an efficient algorithm achieving a regret of 
$\mathcal{O}\left(\sqrt{T\log |\mathcal{G}|}+\sqrt{TK\log |\mathcal{H}|}\right)$


For other cases such as $|\mathcal{G}|$ and $|\mathcal{H}|$ are infinite but with some other assumptions, we can divide them into small partitions and apply suitable algorithms as $alg_G$ and $alg_H$ to obtain the regret bound.




There were some empirical studies on the possibility of evolving the network structures over different tasks to do lifelong learning (Rusu et al., 2016; Lee et al., 2017). For theoretical studies, we, and most prior works as well, focused on learning with fixed architectures. That is, the representation and the predictor spaces are fixed before receiving samples. 



