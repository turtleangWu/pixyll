---
layout: post
title: follow-the-regularized-leader
date: 2018-10-23 12:00
summary: Follow the Regularized Leader (FTRL) 是解 Online Convex Optimization 問題非常常用的方法。以下將簡單介紹 Online Convex Optimization (OCO)，並從 Follow the Leader 的角度解釋為什麼需要 Regularizer。
categories: Online-Learning
author: Yi-Shan Wu
visible: True
---

# Online Convex Optimization (OCO)
在《機器學習問題–Convex Problems(1)》中有稍微介紹過什麼是 convex problem，也就是這個問題的 prediction set 是一個  convex set ，且 loss functions 是 convex function。那篇文章中說我們會碰到的很多問題都是 convex problem，而且幾乎可以說，現在大家會解的問題也幾乎只有 convex function。讀者可能會質疑：「但是曾經看到一些論文有關 non-convex function 啊？」不過事實是，許多解 non-convex 的方法也是奠基於對 convex problem 的分析，因此最重要的還是 convex function。
而在 online learning 的世界也是。很多時候遇到的問題其實都可以是一個標準 Online Convex Optimization 的形式樣子。

顧名思義，在 OCO 中 $S$ 是 convex set，且對於所有的 $t\in [T]$，loss function $\ell_t$ 都是 convex function。那這個 loss function 為什麼有下標？而且為什麼這裡不是像《Online learning — introduction (1)》會收到一個標準答案 $y_t\in \mathcal{Y}$ ，而是收到一個 loss function？其實這是一樣的。如果現在 loss 是像 introduction 一樣是 $\ell:\mathcal{S}\times\mathcal{Y}\rightarrow \mathbb{R}$  ，可以發現當在時間  $t$ 收到標準答案 $y_t$ 後，loss function $\ell(p_t,y_t)$ 也就變成一個單變數函數：$\ell_t(p_t)=\ell(p_t,y_t)$。所以收到標準答案，跟收到 loss function 是一樣的。而 regret 可以寫成

\[Regret_T(u)=\sum\limits_{t=1}^T \ell_t(p_t)-\ell_t(u)  --(1)\]

<center><img src="/images/online/online-6.png" width="120" height="100" /></center>

顧名思義，在 OCO 中 $S$ 是 convex set，且對於所有的 $t\in [T]$，loss function $\ell_t$ 都是 convex function。那這個 loss function 為什麼有下標？而且為什麼這裡不是像《Online learning — introduction (1)》會收到一個標準答案 $y_t\in \mathcal{Y}$ ，而是收到一個 loss function？其實這是一樣的。如果現在 loss 是像 introduction 一樣是 $\ell:\mathcal{S}\times\mathcal{Y}\rightarrow \mathbb{R}$  ，可以發現當在時間  $t$ 收到標準答案 $y_t$ 後，loss function $\ell(p_t,y_t)$ 也就變成一個單變數函數：$\ell_t(p_t)=\ell(p_t,y_t)$。所以收到標準答案，跟收到 loss function 是一樣的。而 regret 可以寫成

\[Regret_T(u)=\sum\limits_{t=1}^T \ell_t(p_t)-\ell_t(u)  --(1)\]

寫 $Regret_T(u)$ 表示如果以 $u$ 為比較對象，玩了 $T$ 步後的差異。而如果以《Online learning — introduction (2)》中的 regret 的定義，這個 $u$ 其實就是

\[u^{*}=arg\min_{u\in \mathcal{S}}\sum\limits_{t=1}^T\ell_t(u)\]

而 (1) 式的 $u$ 也包含了當 $u=u^{*}$ 的狀況，只是這樣寫起來比較 general 一點，表示以任何人當比較對象，這個 $Regret_T(u)$ 都成立。

# Follow the Leader (FTL)

做 online convex optimization 問題，關鍵都在怎麼在每一步做出好的預測，使得每一步的 regret 都不會太大。因為在時間 $t$ ，我最多就只有 $latex \ell_1,\ell_2,\cdots \ell_{t-1}$ 那麼多資訊，因此一個很直覺的預測就是找一個 $p_t$ ：

\[p_t=arg\min_{p\in \mathcal{S}}\sum\limits_{\tau=1}^{t-1}\ell_{\tau}(p)  --(2)\]

也就是看看那個 $p$ 對 $t-1$ 步前的 total loss 看起來是最好的。這個方法乍看之下超棒，不過看看下面這個例子會發現，竟然發生悲劇了！

### Example( by reference [1])

令 $S=[-1,1]\subset \mathbb{R}$ 是一個 convex set ，且 $\ell_t(p)=z_t p$ 是 linear function ，其中

\[z_t=\left\{\begin{array}{cc}-0.5 & \mbox{if }t=1\\ 1&\mbox{ if }t \mbox{ is even}\\ -1& \mbox{if }t>1 \mbox{ and } t\mbox{ is odd} \end{array}\right.\]

很明顯的，如果跑 FTL ，那麼當時間是奇數時， $p_t=1$，反之是 $p_t=-1$。但其實最好的後見之明，是 $p^*=0$。這下子 FTL 的 regret 就是 $ \mathcal{O}(T)$ 了！

由這個例子知道，單純用 FTL 是不可行的，會被 adversary 騙，傻傻的在 $\pm 1$ 之間橫衝直撞。因此大家改用另一個演算法 Follow the Regularized Leader，跟 Follow the Leader 很像，不過在 (2) 式的 objective function 中要再多加 Regularizer，以下將說明為什麼加了 regularizer，以及要加怎麼樣的 regularizer 後，我們就不會像 FTL 一樣預測值在每一回合差異都很大，好像在瞎猜、橫衝直撞。

# Follow the Regularized Leader (FTRL)

FTRL 其實有很多種解釋方式，以下介紹的這一種應該是需要最少背景知識的，雖然不是很精確。1. 2. 3. 的標號表示這個方法直覺上蠻合理的一些關鍵。

### 1. 希望可以偷看下個時間的 loss function

前面提到，在時間 $t$ ，我最多就只有 $\ell_1,\ell_2,\cdots \ell_{t-1}$ 那麼多資訊，但是其實我超希望可以偷看 $\ell_t$ 是什麼，這樣我就可以選到對前 $t$ 步最好的選擇﹝其實也就是 $p_{t+1}$，這用 $p_{t+1}$的定義和數學歸納法就可以證明了﹞。

> Lemma: 令 $p_1,p_2,\cdots$ 為 FTL 演算法產生的預測序列，則 $\forall u\in S$，
> \[\sum\limits_{t=1}^T \ell_t(p_{t+1})\leq \sum\limits_{t=1}^T\ell_t(u)\]

#### Proof: 用數學歸納法

當 $T=1$ 時，根據定義可以知道是對的。
假設到 $T-1$ 時都對，那麼 $\forall u\in S$，
\[\sum\limits_{t=1}^{T-1}\ell_t(p_{t+1})+\ell_T(p_{T+1})\leq \sum\limits_{t=1}^{T-1}\ell_t(u)+\ell_T(p_{T+1})\]

右式對所有 $ u$ 都對，所以當然包含 $ u=p_{T+1}$ 的時候，因此

\[\Rightarrow \sum\limits_{t=1}^{T}\ell_t(p_{t+1})\leq \sum\limits_{t=1}^{T-1}\ell_t(u)+\ell_T(p_{T+1})=\sum\limits_{t=1}^T\ell_t(p_{T+1})\]

最左式是說，當時間 $t$ ，我就選 $p_{t+1}$，每一步都是根據這種規則選；而最右式是說，不管哪一步，我都選擇 $p_{T+1}$ 這個固定的後見之明，而它也正好是

$arg\min_u \sum\limits_{t=1}^T \ell_t(u)$

而這個不等式既然對於最好的後見之明是對的，那麼當然對於所有 $u\in \mathcal{S}$ 都是對的，所以就證完了。

這個證明告訴我們，每一步都都偷看 $\ell_t$ 然後選 $p_{t+1}$ ，其實比一直都玩任何單一的後見之明還好。但是實際上不知道 $\ell_t$ 的話我該怎麼辦？就猜吧！怎麼猜呢？首先，因為所有的 loss function 都是 convex function，所以我當然是猜一個 convex function，先令他叫做 $R:\mathcal{S}\rightarrow \mathbb{R}$。也就是現在在時間 $t$，我希望猜一個 $R(p)$ ，使得下面兩式很像

\[\tilde{p}_t=arg\min_p\sum\limits_{\tau=1}^{t-1}\ell_{\tau}(p)+R(p) -- (3)\]

\[p_{t+1}=arg\min_{p}\sum\limits_{\tau=1}^{t-1}\ell_{\tau}(p)+\ell_t(p)  -- (4)\]

(3)、(4) 要求的是 argmin 要很接近，這點蠻比較不 trivial 的，因為就算兩式右邊的函數值差不多，$\tilde{p}_t$ 也不一定會跟 $p_{t+1}$ 接近。如圖

<center><img src="/images/online/ftrl.png" width="150" height="140" /></center>
