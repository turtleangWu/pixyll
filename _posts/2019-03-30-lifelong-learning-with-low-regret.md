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

To model that different tasks are related, we consider that 


>There is some good representation which is shared by all tasks, but they are different as each requires a different predictor on top of the representation to form a classifier.

In this paper, we consider the following setting: 

# Notations and Settings

* representation space : $\mathcal{G}$ of size $G$ (usually very large)
* predictor space : $\mathcal{H}$ of size $H$
  * to be simple, we consider both $G$ and $H$ are finite in this post
* number of tasks : $K$
* number of samples in each task : $T_k, \forall k\in [K]$ and $\sum T_k =T$
* loss functions are $[0,1]$-valued

Both the tasks and their samples arrive sequentially so that in each time step, we only see a sample from one single task. For each task $k$ and each step $s$ in it, we need to choose a representation $g_{k,s}$ and an accompanying predictor $h_{k,s}$  which jointly provide a decision(classifier) for us. After making this decision, we suffer some loss $\ell_{k,s}(g_{k,s}, h_{k,s})$ according to loss function $\ell_{k,s}$, receive some feedback information, and then proceed to the next iteration.

To measure the performance of a learning algorithm, different settings have their own natural choices. Since the samples of each task arrive one after one, an often adopted measure is the **regret**. Furthermore, to capture the assumption above, we measure the regret by comparing against an offline algorithm which must use a fixed representation for all the tasks but is allowed to use different predictors for different tasks.

$$ 
Regret(T)=\mathbb{E}\left[ \sum\limits_{k,s}\ell_{k,s}(g_{k,s}, h_{k,s})-\min_{g, h_1, \cdots, h_K}\sum\limits_{k,s}\ell_{k,s}(g, h_k) \right] -- (1)
$$

## Example


Before going to details, here we give a simple example to show possible advantages of lifelong learning over relearning tasks.
For the case that $G$ and $H$ are finite but the loss functions are arbitrary, we provide an efficient algorithm achieving a regret of 
$\mathcal{O}\left(\sqrt{T\log G}+\sqrt{TK\log H}\right)$,
while relearning the representation results in regret of


\[\sum\limits_{k=1}^{K}\mathcal{O}\left(\sqrt{T_{k}\log G}+\sqrt{T_{k}\log H}\right)\leq\mathcal{O}\left(\sqrt{KT\log G}+\sqrt{KT\log H}\right) \]


First of all, our bound prevents the number of tasks from affecting the learning of representations. That is to say, the regret of learning the representations doesn't grow with the number of tasks (for a fixed $T$). Since $G$ is usually large, this benefit makes our bound attractive for large $K$.

Moreover, as learning the representations is typically much more costly than learning predictors in lifelong learning, if under some conditions it is possible to identify the best representation $g^{\*}$ for all tasks at some step $t<T$, this would allow us to learn new tasks faster by saving the time for learning the representation.

# First Challenge -- construct losses

In learning problems, we always guide the learning by losses. However, here the losses $\ell_{k,s}(g_{k,s}, h_{k,s})$ depend on both the representation and the predictor. This makes learning harder. If we already know what the best representation $g^{\*}$ is, it remains to learn predictors for each task. However, how can we estimate how good a representation is when **a good representation may look bad if we choose a bad predictor to go with it**? A sensible choice seems to be accompanying it with its **best predictor in a task**. Take the full-information adversarial setting for example.


### Naive Choice for Full-Information Adversarial Setting

In full-information setting, the whole loss fuction at each step is revealed. A sensible choice to measure a representation $g$ in task $k$ is $\hat{L}_k (g)$, where

$$ 
\hat{L}_k(g)=\min_h \sum\limits_{s=1}^{T_k} \ell_{k,s}(h, g). 
$$

With $$\hat{L}_k (g)$$, we can update the learning of $g$ at the end of each task accordingly. When starting a new task $k$, choose a $g_k$ to serve every step in the task. Moreover, use the $g_k$ throughout task to learn $h_{k,s}$ at each step.


Everything goes well so far. Nevertheless, the above method only provides us with measurement at the end of tasks. When learning within a task, we do not know what the best predictor of a representation is as **the predictor which looks best so far may turn out to be bad at the end of the task in the adversarial setting**. This is perhaps one reason why Alquier et al. (2017) chose to update their representations only at the end of each task, where the best predictor of each representation in the task is ensured. This consequently requires a large number of tasks in order to have a good regret bound due to less update. 

To achieve our regret bound, we have to construct appropriate loss functions so as to update representations more often (we actually update them at every step). 

## Idea to Update Representations Often

Recall that tasks are related as they share some common representation, but they are different as each requires a different predictor on top of the representation. Therefore, we hope to **learn the representations continuously through time**, while we still have to **relearn predictors for different tasks**. To do that, we would like to decouple the learning of representations from that of predictors, for them to have different loss functions and different learning schedules. For finite representations, we describe our solution via a generic algorithm.

### Algorithm 1

We take $alg_G$ to learn the representation and have it update continuously through time across different tasks. For each possible representation $g$, we have a separate copy of $alg^{(g)}_{H}$ for learning the accompanying predictors. When starting a new task $k$, reset each copy $alg^{(g)}_H$ and redo its learning.


At step $s$ in task $k$, we sample a representation $g_{k,s}$ according to the distribution $G_{k,s}$ of $alg_{G}$, followed by sampling a predictor $h_{k,s}$ according to the distribution $H_{k,s}^{(g_{k,s})}$ of $alg_{H}^{(g_{k,s})}$. The joint action we play is $(g_{k,s}, h_{k,s})$ and suffer the loss $\ell_{k,s}(g_{k,s}, h_{k,s})$. Then we update the distribution of $alg_{G}$ using some loss function $$\tilde{\ell}_{k,s}(g)$$ defined on representation $g$ while update $$alg_{H}^{(g)}$$  for each $g$ using some loss function $$\hat{\ell}_{k,s}(g,h)$$ defined on $h$ with respect to a specific $g$. The loss functions would be specified later for different settings accordingly.

### Full-Information Adversarial Setting

Recall that in full-information setting, the whole loss fuction at each step is revealed. Here we define $\tilde{\ell}_{k,s}(g)$ as


$$\tilde{\ell}_{k,s}(g)=\mathbb{E}_{h\sim H_{k,s}^{(g)}}\left[\ell_{k,s}(g,h) \right] $$


and define the loss $$\hat{\ell}_{k,s}(g,h) = \ell_{k,s}\left( g,h \right)$$ to be the loss function on predictors. That is, the loss of $g$ at each step is defined to be the average loss of $g$ with its predictors, while the loss of predictors should be defined with respect to a specific $g$. With this algorithm, we have the following theorem.

>**Theorem 1 :**
>Suppose the $t$-step regret bounds of $alg_G$ and $alg_H$ are $reg_{G}(t)$ and $reg_{H}(t)$, respectively. Then the $T$-step regret bound of our algorithm with the defined losses is at most $reg_G\left(T\right) +\sum\limits_{k=1}^{K} reg_H\left(T_k \right)$.

Now for full-information adversarial cases, if we use multiplicative update (MU) algorithm as $alg_G$ and $alg_H$ we can obtain our result for above example:


>**Corollary :**
>For the case that $G$ and $H$ are finite but the loss functions are arbitrary, we provide an efficient algorithm achieving a regret of 
>$\mathcal{O}\left(\sqrt{T\log G}+\sqrt{TK\log H}\right)$


For other cases such as $G$ and $H$ are infinite but with some other assumptions, we can divide them into small partitions and apply suitable algorithms as $alg_G$ and $alg_H$ to obtain the regret bound. You can check the paper for further details.

### Bandit Adversarial Setting

Here we consider the bandit setting, in which the feedback information is the loss value $\ell_{k,s}(g_{k,s}, h_{k,s})$ of our action $(g_{k,s}, h_{k,s})$, instead of the whole loss function $\ell_{k,s}\left(\cdot\right)$. $G$ and $H$ are again set to be finite. This is obviously harder than full-information setting that we do not have the whole loss function to guide the learning.  We would like to see if the above algorithm can also deal with bandit adversarial setting.


Following previous works for bandit setting, our approach is to **construct appropriate estimators of the true loss functions**, $$\bar{\ell}_{k,s}$$, which would be specified later, and feed the estimator to update appropriate full-information algorithms. An appropriate estimator should be unbiased. That is, conditioned on all previous randomness, the expected value of it is exactly the true loss function. A natural estimator for $\ell_{k,s}\left( g, h \right)$ is the following:

$$
\bar{\ell}_{k,s}\left(g,h\right)=\frac{\ell_{k,s}(g,h)}{G_{k,s}(g)\cdot H_{k,s}^{(g)}(h)}\mathbf{1}_{g=g_{k,s},h=h_{k,s}},
$$


where $G_{k,s}(g)$ and $H_{k,s}^{(g)}(h)$ denote the probabilities of choosing $g$ and $h$, respectively. It is not hard to check that $$\bar{\ell}_{k,s}$$ is an unbiased estimator of $\ell_{k,s}$ for any $g$ and $h$.


# Second Challenge -- Low Sampling Probability in Bandit Setting

Nevertheless, in bandit setting, a problem is how to make sure that all $g$ would be sampled often. This is because if a representation is chosen with a low probability, we rarely has the chance to receive the needed feedbacks to learn its accompanying predictors well. Also, without learning the predictors well, we cannot choose the representations appropriately. Moreover, low sampling probability of $g$  could results in large $$\bar{\ell}_{k,s}$$ and consequently bad regret bound.


### Algorithm 2

Our bandit algorithm is basically modified from our full-information algorithm in Algorithm 1, by using the estimator $$\bar{\ell}_{k,s}$$ in place of the true loss $$\ell_{k,s}$$ for updates, but also taking the above issue into account. To address this issue, a possible solution is to add an **additional exploration probability** to the distribution of representations, so that $G_{k,s}(g)$ is large enough for each $g$. Note that this part is not put in the final version of our paper. However, it is still an idea worth mentioned. 

To be specified, we use $$\hat{\ell}_{k,s}(g,h) = \bar{\ell}_{k,s}\left( g,h \right)$$ to update the distribution of predictors, $$\mathcal{H}_{k,s}^{(g)}(h) $$,  according to the MU algorithm. Moreover, to update the distribution of representations, we first feed the loss function

$$\tilde{\ell}_{k,s}(g)=\mathbb{E}_{h\sim H_{k,s}^{(g)}}\left[ \bar{\ell}_{k,s}(g,h) \right]=\frac{\ell_{k,s}(g,h)}{G_{k,s}(g)}\mathbf{1}_{g=g_{k,s}} $$

to update some distribution $$q_{k,s}(g)$$ according to the MU algorithm. In addition, we introduce an additional exploration probability $\rho$ and update $$G_{k,s}(g) = \rho \cdot (1/G) + (1 − \rho) \cdot q_{k,s}(g).$$ The algorithm is summarized below.

<center class="half">
  <img src="/images/lifelong/Algorithm2.png" width="600" height="450" />
</center>

This algorithm results in the following theorem. 

>**Theorem 2 :** 
>For bandit adversarial setting with finite $G$ and $H$, our algorithm achieves a regret of
>$\mathcal{O}\left(\sqrt{TG\log G}+ (T^{2}KGH \log H)^{1/3}\right)$


The regret bound has the order of $2/3$ dependency on $T$. However, we provides a lower bound for the problem for only the order of $1/2$ dependency on $T$. Therefore, Algorithm 2 is obviously not good enough.

>**Theorem 3 :**
>The problem with finite $G$ and $H$ and arbitrary loss functions in the bandit setting has a regret lower bound of
>$\Omega \left( \sqrt{TGH} + \sqrt{TKH} \right) $

### Algorithm 3

we take a different approach, by reducing our problem to the following **"experts over actions" problem**. In this new problem, there is a set $$\mathcal{G} \times \mathcal{H}^K$$ of experts. Each expert is indexed by some $(g, \vec{h})$, with $g\in \mathcal{G}$ and $\vec{h} = \left( h_1, h_2,\cdots , h_K \right)$, who in every step $s$ of task $k$ plays the action $\left( g, h_k\right)$, which has the loss value $\ell_{k,s}\left( g,h_k \right)$. Now what an online algorithm can do at each step is to choose an expert and play his/her action, and the regret is measured against the total loss of the best expert, which in fact is the same as the regret defined in Eq.(1). Therefore, we can run the EXP3 algorithm (Auer et al. (2002b)) on the experts and apply its regret analysis. This algorithm obtained a better regret bound.

>**Theorem 2 :** 
>For the problem with finite $G$ and $H$ and arbitrary loss functions, our bandit algorithm achieves a regret of
>$\mathcal{O}\left(\sqrt{TGH\log (GH^K)}\right)$

There is an apparent efficiency issue for maintaining $GH^K$ experts. We avoid this problem such that it suffices to be able to sample from the distribution of $GH$ actions played by the experts at each step. For more details, please refer to our paper.

# Third Challenge -- Stochastic Setting

Consider that for each task $k$, there is some fixed but unknown distribution that the loss function $\ell_{k,s}$ is sampled i.i.d. from it in a task, with mean $\mu_k(g,h)$ for any $(g,h)$. To measure how good a representation $g$ is in task $k$, we let $$\mu_{k}(g) = \min_h \mu_{k}(g,h)$$. Also, we can define

$$
\begin{eqnarray}
\Delta = \min_k \min_{g \ne g^{*}} \left(\mu_k(g) - \mu_k(g^{*})\right)  \mbox{  and  } \\
\Delta_* = \min_k \min_{h \ne h^{*}_k} \left(\mu_k(g^{*},h) - \mu_k(g^{*})\right). 
\end{eqnarray}
$$


To make the tasks related, we assume that

>The best representation $g$ in every task is the same, denoted as $g^{\*}$.

Following previous works for the stochastic setting, we hope that $g^{*}$ can be determined within a small number of iterations. In traditional single task problems, we use follow-the-leader algorithm and UCB (Auer et al. (2002a)) for full-information and bandit setting, respectively.

UCB guarantees that an arm with gap $\Delta$,  is likely to be distinguished for about $1/\Delta^2$ iterations, and each such iteration contributes $\Delta$ to the total regret. The single task problem of total $T$ steps with $GH$ arms achieves regret of 

$$Regret(T)=\frac{GH\log T}{\Delta} $$

However, the standard regret analysis of those algorithms rely crucially on the assumption that the mean of each arm’s loss does not change over time. In our case, the mean loss of a representation $g$ may keep changing when going into new tasks. The previous methods apparently fail. Obviously, it requires to design a new algorithm for this problem. 

If we simply relearn all tasks by UCB, we obtain the regret of

$$Regret(T)=\frac{KGH\log T}{\Delta} $$

How can we do better than that?








Algorithm Our algorithm works in two phases.


In the exploration phase, we run our adversarial algorithm until some iteration $\hat{T}$ when there is some representation $\hat{g}$ which dominates others. Then we enter the exploitation phase in which we always choose the representation $\hat{g}$. As there is no representation to learn, the problem of learning each task then reduces to the traditional single task problem. Therefore, we can use \underline{follow the leader} (UCB) algorithm for \underline{full-information} (bandit) setting to learn the accompanying predictors.
