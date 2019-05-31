---
layout: post
title: Regret in Online Stochastic Settings
date: 2019-05-29 12:00
summary: Providing a sublinear regret bound is the primary method to show that an online learning algorithm actually works. Although most works focus on adversarial settings, there are still a lot of interests in stochastic cases. Here are some frequently used arguments to obtain low regret bound for stochastic settings.
categories: Online-Learning
author: Yi-Shan Wu
visible: True
---

# Introduction

Providing a sublinear regret bound is the primary method to show that an online learning algorithm actually works. Although most works focus on worst-case settings (adversarial settings), there are still a lot of interests in stochastic cases. Here are some frequently used arguments to obtain low regret bound for stochastic settings.

In the most basic formulation of $K$-armed stochastic bandit setting problem, at each step $t$, we play an arm $I_t\in [K]$ and receive the loss $\ell_t(I_t)$, where $\ell_t$ is the loss function at step $t$. The losses of playing the arm $i$, are sampled i.i.d. from some unknown mean $\mu_i$. That is, for all $t\in \mathbb{N}$ and $i\in [K]$, $\mathbb{E}\left[\ell_t(i)\right]=\mu_i$. Furthermore, let $i^{\*}=arg\min_i \mu_i$ be the unique best arm which is considerably better than other arms with some gap $\Delta_i=\mu_i-\mu_{i^\*}$, respectively.

The pseudo regret is often used for stochastic setting. Therefore, for a total $T$ steps game, the total regret is defined to be

$$ \bar{Reg_T}=\mathbb{E}\left[\sum\limits_{t=1}^T \ell_t(I_t)-\ell_t(i^{\*}) \right]$$

which can be further written as

$$ \sum\limits_{i\neq i^{\*} }\Delta_i\mathbb{E}\left[N_T(i)\right]  $$

where $N_T(i)$ is the number of times we play arm $i$ within $T$ steps.


# Ideas

Since the best arm is better than other arms with certain gaps, if it is possible that we can identify the best arm in short time, then 








# Reference

1. [Finite-time Analysis of the Multiarmed Bandit Problem](https://link.springer.com/article/10.1023/A:1013689704352)
1. [The Nonstochastic Multiarmed Bandit Problem](https://epubs.siam.org/doi/abs/10.1137/S0097539701398375?casa_token=zXo4I7PhVt0AAAAA:eImrtCW6kfJqiLcIzNRUCpoedDQOCxJ8VQYMbHXB4t9Ca9jR7Gvxf6ONMP2O8S3tvo_K0VqRi3dU)
1. [On the optimality of the Hedge algorithm in the stochastic regime](https://arxiv.org/abs/1809.01382)
1. [One Practical Algorithm for Both Stochastic and Adversarial Bandits (ICML 2014)](http://proceedings.mlr.press/v32/seldinb14.html)
1. [An improved parametrization and analysis of the EXP3++ algorithm for stochastic and adversarial bandits (COLT 2017)](https://arxiv.org/abs/1702.06103)



