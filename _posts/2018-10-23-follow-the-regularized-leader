---
layout: post
title: follow-the-regularized-leader
date: 2018-10-23 12:00
summary: Follow the Regularized Leader (FTRL) 是解 Online Convex Optimization 問題非常常用的方法。以下將簡單介紹 Online Convex Optimization (OCO)，並從 Follow the Leader 的角度解釋為什麼需要 Regularizer。
categories: Online-Learning
---

###Online Convex Optimization (OCO)
在《機器學習問題–Convex Problems(1)》中有稍微介紹過什麼是 convex problem，也就是這個問題的 prediction set 是一個  convex set ，且 loss functions 是 convex function。那篇文章中說我們會碰到的很多問題都是 convex problem，而且幾乎可以說，現在大家會解的問題也幾乎只有 convex function。讀者可能會質疑：「但是曾經看到一些論文有關 non-convex function 啊？」不過事實是，許多解 non-convex 的方法也是奠基於對 convex problem 的分析，因此最重要的還是 convex function。
而在 online learning 的世界也是。很多時候遇到的問題其實都可以是一個標準 Online Convex Optimization 的形式樣子。

顧名思義，在 OCO 中 $S$ 是 convex set，且對於所有的 $ t\in [T]$，loss function $ \ell_t$ 都是 convex function。那這個 loss function 為什麼有下標？而且為什麼這裡不是像《Online learning — introduction (1)》會收到一個標準答案 $y_t\in \mathcal{Y}$ ，而是收到一個 loss function？其實這是一樣的。如果現在 loss 是像 introduction 一樣是 $\ell:\mathcal{S}\times\mathcal{Y}\rightarrow \mathbb{R}$  ，可以發現當在時間  $t$ 收到標準答案 $y_t$ 後，loss function $\ell(p_t,y_t)$ 也就變成一個單變數函數：$\ell_t(p_t)=\ell(p_t,y_t)$。所以收到標準答案，跟收到 loss function 是一樣的。而 regret 可以寫成

\[Regret_T(u)=\sum\limits_{t=1}^T \ell_t(p_t)-\ell_t(u)  --(1)\]
