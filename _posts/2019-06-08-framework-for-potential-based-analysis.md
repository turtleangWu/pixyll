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

This algorithm comes from the usual properties of convex analysis. Recall that the convex conjugate of a function $f:\mathbb{R}^K\rightarrow \mathbb{R}$ is

$$f^*(y)=max_{x\in \mathbb{R}^K}\left\{\langle x,y \rangle -f(x) \right\}. $$

In online learning problems, our decision sets are often constrained. Therefore, for constrained convex set $A\subset \mathbb{R}^K$, we define

$$(f+\mathcal{I}_A)^*(y)=max_{x\in A}\left\{\langle x,y \rangle -f(x) \right\}. $$

Furthermore, by standard results from convex analysis,  for differentiable and convex $f$ with invertible gradient $(\nabla f)^{-1}$ it holds that

$$\nabla(f+\mathcal{I}_A)^*(y)=arg\max_{x\in A}\left\{\langle x,y \rangle -f(x) \right\}\in A. $$





# Reference
1. [The Nonstochastic Multiarmed Bandit Problem](https://epubs.siam.org/doi/abs/10.1137/S0097539701398375?casa_token=zXo4I7PhVt0AAAAA:eImrtCW6kfJqiLcIzNRUCpoedDQOCxJ8VQYMbHXB4t9Ca9jR7Gvxf6ONMP2O8S3tvo_K0VqRi3dU)
1. [Online Linear Optimization via Smoothing](http://www.jmlr.org/proceedings/papers/v35/abernethy14.pdf)

