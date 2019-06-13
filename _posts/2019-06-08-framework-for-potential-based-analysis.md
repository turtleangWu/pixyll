---
layout: post
title: Analysis of Gradient-Based Prediction Algorithm for Multi-Armed Bandit Problems
date: 2019-06-08 12:00
summary: Here we give a general framework to analyze regret bound running with follow-the-regularized-leader type algorithms.
categories: Online-Learning
author: Yi-Shan Wu
visible: True
---

# Introduction

Roughly Speaking, online algorithms can often be written in Online Mirror Descent (OMD) or Follow-the-Regularized-Leader (FTRL) type. Since the posts in the future would mostly be in FTRL type, here we provide a general framework for analyzing regret. Actually, there is no big difference in their analysis, but sometimes one is easier than the other. Anyway, I'm just writing this post for my next one.

In the most basic formulation of $K$-armed bandit problem, at each step $t$, we play an arm $I_t\in [K]$ according to some distribution $w_t$ over arms, and receive the loss $\ell_t(I_t)$, where $\ell_t$ is the loss function at step $t$. The losses, which could be either adversarial or stochastic, are determined by environment. In bandit problems, people usually construct unbiased losses for update. That is, let 

$$\hat{\ell}_t(i)=\frac{\ell_t(i)}{w_t(i)}\mathbf{1}_{i=I_t}, \forall i\in [K] $$

, where $\ell_t(i)=\ell_t(I_t)$ as $i=I_t$; otherwise, $\ell_i$ (and thus $\ell_t(i)$) is zero.


# Potential function techniques

This algorithm is proposed in $(ref.2)$ and you can find some of theorems and proofs in the paper. Recall that the convex conjugate of a function $f:\mathbb{R}^K\rightarrow \mathbb{R}$ is

$$f^*(y)=max_{x\in \mathbb{R}^K}\left\{\langle x,y \rangle -f(x) \right\}. $$

In online learning problems, our decision sets are often constrained (probability simplex in this post). Therefore, for constrained convex set $A\subset \mathbb{R}^K$, we define

$$(f+\mathcal{I}_A)^*(y)=max_{x\in A}\left\{\langle x,y \rangle -f(x) \right\}. $$

Furthermore, by standard results from convex analysis,  for differentiable and convex $f$ with invertible gradient $(\nabla f)^{-1}$ it holds that

$$\nabla(f+\mathcal{I}_A)^*(y)=arg\max_{x\in A}\left\{\langle x,y \rangle -f(x) \right\}\in A. $$

Therefore, authors design the GBPA algorithm as

<center><img src="/images/online/GBPA.png" width="630" height="300" /></center>

First of all, the function $\psi_t(w)$ serves both as the regularizer in FTRL algorithm and as the specific function for its conjugate $\Phi_t$. The regularizer $\psi_t(w)$ plays a central role in the algorithm for controlling the ''smoothness''. Also, the algorithm is implementable only when the regularizer is well chosen. In general, $\psi_t(w)$ can be different at different time $t$, although in some cases a good regret bound can be obtained by choosing the same regularizer for all $t$. 


# Regret Analysis

It is known that the (expected) regret can be written as

$$\mathbb{E}\left[\sum\limits_{t=1}^T\ell_t(I_t)-\sum\limits_{t=1}^T\ell_t(i_T^*)\right],$$

where $i_T^*$ is defined as the best arm in expectation in hindsight. Since 

$$ \ell_t(I_t)= \langle w_t, \hat{\ell}_t \rangle =  \langle \nabla\Phi_t(-\hat{L}_{t-1}), \hat{L}_t-\hat{L}_{t-1} \rangle, $$

where the last equality comes from the definition of $w_t$. By the definition of Bregman divergence, it also equals

$$ \Phi_t(-\hat{L}_{t-1})-\Phi_t(-\hat{L}_t)+D_{\Phi_t}(-\hat{L}_t, -\hat{L}_{t-1}). $$

The regret becomes

$$\mathbb{E}\left[\sum\limits_{t=1}^T\big( \Phi_t(-\hat{L}_{t-1})-\Phi_t(-\hat{L}_t) -\hat{\ell}_t(i_T^*)\big) +\sum\limits_{t=1}^T\big(D_{\Phi_t}(-\hat{L}_t, -\hat{L}_{t-1})\big) \right]. $$

Here we introduce a key lemma in this post.

-> $$\sum\limits_{t=1}^T\big( \Phi_t(-\hat{L}_{t-1})-\Phi_t(-\hat{L}_t) -\hat{\ell}_t(i_T^*)\big)\leq \sum\limits_{t=1}^T-\psi_t(w_t)+\psi_t(w_{t+1})$$

With the key lemma, the regret can be bounded by 

$$\mathbb{E}\left[\sum\limits_{t=1}^T \big(-\psi_t(w_t)+\psi_t(w_{t+1})\big) +\sum\limits_{t=1}^T\big(D_{\Phi_t}(-\hat{L}_t, -\hat{L}_{t-1})\big) \right] $$

Thus, we can bound the two summation adapted to different cases.

# Proof of the key lemma

Since 

$$\Phi_t(-\hat{L}_{t-1}) = \langle w_t, -\hat{L}_{t-1}\rangle -\psi_t(w_t), \mbox{and}$$

$$\Phi_t(-\hat{L}_{t}) = \max_w \langle w, -\hat{L}_{t}\rangle - \psi_t(w)\geq  \langle w_{t+1}, -\hat{L}_{t}\rangle -\psi_t(w_{t+1}),$$

we have

$$\sum\limits_{t=1}^T\left(\Phi_t(-\hat{L}_{t-1})-\Phi_t(-\hat{L}_t)\right) \leq  \sum\limits_{t=1}^T \left(\langle w_t, -\hat{L}_{t-1} \rangle - \langle w_{t+1}, -\hat{L}_t \rangle \right) + \sum\limits_{t=1}^T \big(-\psi_t(w_t)+\psi_t(w_{t+1})$$

$$=\langle w_{T+1}, \hat{L}_T\rangle + \sum\limits_{t=1}^T\big(-\psi_t(w_t)+\psi_t(w_{t+1})\big)$$

$$w_{T+1}$$ can be naturally chosen as $$e_{i_T^{*}}$$. Thus, $$\langle e_{i_T^*}, \hat{L}_T\rangle$$  cancels out the  summation $$\sum\limits_{t=1}^T -\hat{\ell}_t(i_T^*)$$. 



# Reference
1. [The Nonstochastic Multiarmed Bandit Problem](https://epubs.siam.org/doi/abs/10.1137/S0097539701398375?casa_token=zXo4I7PhVt0AAAAA:eImrtCW6kfJqiLcIzNRUCpoedDQOCxJ8VQYMbHXB4t9Ca9jR7Gvxf6ONMP2O8S3tvo_K0VqRi3dU)
1. [Online Linear Optimization via Smoothing](http://www.jmlr.org/proceedings/papers/v35/abernethy14.pdf)

