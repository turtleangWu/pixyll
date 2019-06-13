---
layout: post
title: Studies for EXP3 Algorithm in Stochastic Regime
date: 2019-06-10 12:00
summary: We would like to see how EXP3 algorithm performs in stochastic settings.
categories: Online-Learning
author: Yi-Shan Wu
visible: True
---

# Introduction

EXP3 algorithm is proposed in 2002$(ref.2)$. It is shown that EXP3 algorithm can achieve $$\mathcal{O}\left(\sqrt{KT\log K}\right)$$, where $K$ is the number of arms in expectation.

In the most basic formulation of $K$-armed stochastic bandit setting problem, at each step $t$, we play an arm $I_t\in [K]$ and receive the loss $\ell_t(I_t)$, where $\ell_t$ is the loss function at step $t$. The losses of playing the arm $i$, are sampled i.i.d. from some unknown mean $\mu_i$. That is, for all $t\in \mathbb{N}$ and $i\in [K]$, $\mathbb{E}\left[\ell_t(i)\right]=\mu_i$. Furthermore, let $i^{\*}=arg\min_i \mu_i$ be the unique best arm which is considerably better than other arms with some gap $\Delta_i=\mu_i-\mu_{i^\*}$, respectively.

The pseudo regret is often used for stochastic setting. Therefore, for a total $T$ steps game, the total regret is defined to be

$$ \bar{Reg_T}=\mathbb{E}\left[\sum\limits_{t=1}^T \ell_t(I_t)-\ell_t(i^{*}) \right]$$

which can be further written as

$$ \sum\limits_{i\neq i^{*} }\Delta_i\mathbb{E}\left[N_T(i)\right]  $$

where $N_T(i)$ is the number of times we play arm $i$ within $T$ steps.







# Reference

1. [Finite-time Analysis of the Multiarmed Bandit Problem](https://link.springer.com/article/10.1023/A:1013689704352)
1. [The Nonstochastic Multiarmed Bandit Problem](https://epubs.siam.org/doi/abs/10.1137/S0097539701398375?casa_token=zXo4I7PhVt0AAAAA:eImrtCW6kfJqiLcIzNRUCpoedDQOCxJ8VQYMbHXB4t9Ca9jR7Gvxf6ONMP2O8S3tvo_K0VqRi3dU)
1. [PAC-Bayesian Inequalities for Martingales](https://arxiv.org/pdf/1110.6886.pdf)
1. [PAC-Bayesian Analysis of Martingales and Multiarmed Bandits](https://arxiv.org/abs/1105.2416)
1. [Evaluation and Analysis of the Performance of the EXP3 Algorithm in Stochastic Environments](http://proceedings.mlr.press/v24/seldin12a/seldin12a.pdf)
1. [One Practical Algorithm for Both Stochastic and Adversarial Bandits (ICML 2014)](http://proceedings.mlr.press/v32/seldinb14.html)
1. [An algorithm with nearly optimal pseudo-regret for both stochastic and adversarial bandits](https://arxiv.org/abs/1605.08722)
1. [An improved parametrization and analysis of the EXP3++ algorithm for stochastic and adversarial bandits (COLT 2017)](https://arxiv.org/abs/1702.06103)
