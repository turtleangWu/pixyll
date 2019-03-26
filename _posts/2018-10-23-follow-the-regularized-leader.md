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

<center><img src="/images/online/online-6.png" width="320" height="200" /></center>

顧名思義，在 OCO 中 $S$ 是 convex set，且對於所有的 $t\in [T]$，loss function $\ell_t$ 都是 convex function。那這個 loss function 為什麼有下標？而且為什麼這裡不是像《Online learning — introduction (1)》會收到一個標準答案 $y_t\in \mathcal{Y}$ ，而是收到一個 loss function？其實這是一樣的。如果現在 loss 是像 introduction 一樣是 $\ell:\mathcal{S}\times\mathcal{Y}\rightarrow \mathbb{R}$  ，可以發現當在時間  $t$ 收到標準答案 $y_t$ 後，loss function $\ell(p_t,y_t)$ 也就變成一個單變數函數：$\ell_t(p_t)=\ell(p_t,y_t)$。所以收到標準答案，跟收到 loss function 是一樣的。而 regret 可以寫成

\[Regret_T(u)=\sum\limits_{t=1}^T \ell_t(p_t)-\ell_t(u)  --(1)\]

寫 $Regret_T(u)$ 表示如果以 $u$ 為比較對象，玩了 $T$ 步後的差異。而如果以《Online learning — introduction (2)》中的 regret 的定義，這個 $u$ 其實就是

\[u^{*}=arg\min_{u\in \mathcal{S}}\sum\limits_{t=1}^T\ell_t(u)\]

而 (1) 式的 $u$ 也包含了當 $u=u^{*}$ 的狀況，只是這樣寫起來比較 general 一點，表示以任何人當比較對象，這個 $Regret_T(u)$ 都成立。

# Follow the Leader (FTL)

做 online convex optimization 問題，關鍵都在怎麼在每一步做出好的預測，使得每一步的 regret 都不會太大。因為在時間 $t$ ，我最多就只有 $latex \ell_1,\ell_2,\cdots \ell_{t-1}$ 那麼多資訊，因此一個很直覺的預測就是找一個 $p_t$ ：

\[p_t=arg\min_{p\in \mathcal{S}}\sum\limits_{\tau=1}^{t-1}\ell_{\tau}(p)  --(2)\]

也就是看看那個 $p$ 對 $t-1$ 步前的 total loss 看起來是最好的。這個方法乍看之下超棒，不過看看下面這個例子會發現，竟然發生悲劇了！

### Example( by reference 1.)

令 $S=[-1,1]\subset \mathbb{R}$ 是一個 convex set ，且 $\ell_t(p)=z_t p$ 是 linear function ，其中

\[
z_t=\left\{\begin{array}{cc}-0.5 & \mbox{if }t=1\\ 1&\mbox{ if }t \mbox{ is even}\\ -1& \mbox{if }t>1 \mbox{ and } t\mbox{ is odd} \end{array}\right.
\]

很明顯的，如果跑 FTL ，那麼當時間是奇數時， $p_t=1$，反之是 $p_t=-1$。但其實最好的後見之明，是 $p^*=0$。這下子 FTL 的 regret 就是 $ \mathcal{O}(T)$ 了！

由這個例子知道，單純用 FTL 是不可行的，會被 adversary 騙，傻傻的在 $\pm 1$ 之間橫衝直撞。因此大家改用另一個演算法 Follow the Regularized Leader，跟 Follow the Leader 很像，不過在 (2) 式的 objective function 中要再多加 Regularizer，以下將說明為什麼加了 regularizer，以及要加怎麼樣的 regularizer 後，我們就不會像 FTL 一樣預測值在每一回合差異都很大，好像在瞎猜、橫衝直撞。

# Follow the Regularized Leader (FTRL)

FTRL 其實有很多種解釋方式，以下介紹的這一種應該是需要最少背景知識的，雖然不是很精確。1. 2. 3. 的標號表示這個方法直覺上蠻合理的一些關鍵。

## 1. 希望可以偷看下個時間的 loss function

前面提到，在時間 $t$ ，我最多就只有 $\ell_1,\ell_2,\cdots \ell_{t-1}$ 那麼多資訊，但是其實我超希望可以偷看 $\ell_t$ 是什麼，這樣我就可以選到對前 $t$ 步最好的選擇﹝其實也就是 $p_{t+1}$，這用 $p_{t+1}$的定義和數學歸納法就可以證明了﹞。

> Lemma: 令 $p_1,p_2,\cdots$ 為 FTL 演算法產生的預測序列，則 $\forall u\in S$，
> \[\sum\limits_{t=1}^T \ell_t(p_{t+1})\leq \sum\limits_{t=1}^T\ell_t(u)\]


### Proof: 用數學歸納法

當 $T=1$ 時，根據定義可以知道是對的。
假設到 $T-1$ 時都對，那麼 $\forall u\in S$，
\[\sum\limits_{t=1}^{T-1}\ell_t(p_{t+1})+\ell_T(p_{T+1})\leq \sum\limits_{t=1}^{T-1}\ell_t(u)+\ell_T(p_{T+1})\]

右式對所有 $ u$ 都對，所以當然包含 $ u=p_{T+1}$ 的時候，因此

\[\Rightarrow \sum\limits_{t=1}^{T}\ell_t(p_{t+1})\leq \sum\limits_{t=1}^{T-1}\ell_t(u)+\ell_T(p_{T+1})=\sum\limits_{t=1}^T\ell_t(p_{T+1})\]

最左式是說，當時間 $t$ ，我就選 $p_{t+1}$，每一步都是根據這種規則選；而最右式是說，不管哪一步，我都選擇 $p_{T+1}$ 這個固定的後見之明，而它也正好是

\[arg\min_u \sum\limits_{t=1}^T \ell_t(u)\]

而這個不等式既然對於最好的後見之明是對的，那麼當然對於所有 $u\in \mathcal{S}$ 都是對的，所以就證完了。

這個證明告訴我們，每一步都都偷看 $\ell_t$ 然後選 $p_{t+1}$ ，其實比一直都玩任何單一的後見之明還好。但是實際上不知道 $\ell_t$ 的話我該怎麼辦？就猜吧！怎麼猜呢？首先，因為所有的 loss function 都是 convex function，所以我當然是猜一個 convex function，先令他叫做 $R:\mathcal{S}\rightarrow \mathbb{R}$。也就是現在在時間 $t$，我希望猜一個 $R(p)$ ，使得下面兩式很像

\[
\tilde{p}_t=arg\min_p\sum\limits_{\tau=1}^{t-1}\ell_{\tau}(p)+R(p) -- (3)
\]

\[p_{t+1}=arg\min_{p}\sum\limits_{\tau=1}^{t-1}\ell_{\tau}(p)+\ell_t(p)  -- (4)\]

(3)、(4) 要求的是 argmin 要很接近，這點蠻比較不 trivial 的，因為就算兩式右邊的函數值差不多，$\tilde{p}_t$ 也不一定會跟 $p_{t+1}$ 接近。如圖

<center><img src="/images/online/ftrl.png" width="300" height="200" /></center>*若藍色是 $latex \ell_t$，那麼無論比它彎曲，還是比它平滑，要達到同樣 y 值對應到的 x 不一樣。*

如果 $R(p)$ 比 $\ell_t(p)$ 還要平滑﹝如黃線﹞，那麼若要達到同樣的函數值，$\tilde{p}_t$ 可能比 $ p_{t+1}$ 還大。但若是選較彎曲的函數﹝如紅線﹞，那麼$\tilde{p}_t$ 可能比 $ p_{t+1}$ 還小。看起來選黃的或選紅的都不好。

所以到目前為止，我只知道要選一個 convex function $R$，而且 $R(p)$ 夠大，大到大概就是 loss 的那個 order 。

## 2. FTRL 的 update 要夠平滑

前面也提到一個問題，如果從 $p_t\rightarrow p_{t+1}$ 改變很大，我們可能就很容易被騙。所以我希望我加了 regularizer 後的 update 方式 ：$ \tilde{p}_t\rightarrow \tilde{p}_{t+1}$ 可以不要變化那麼大。意思就是我比較傾向選彎曲一點的線，這樣 $x$ 值改變一點點 $y$ 值就可以有蠻大的改變。在嚴謹一點的證明中，其實會發現可以有好的 regret bound 的 regularizer 必須要求是 strongly-convex function。

給了一堆不嚴謹的解釋後，筆者只是想引出 FTRL 演算法一個可能的解釋方向。而它的演算法：

> \[\tilde{p}_t=arg\min_p\sum\limits_{\tau=1}^{t-1}\ell_{\tau}(p)+R(p) -- (3)\]
> 其中 $latex R:\mathcal{S}\rightarrow \mathbb{R}$ 是一個 strongly-convex function。

### Analysis of FTRL

有了以上的材料，再加上 convex function 的一些性質，就很容易可以寫下 FTRL 的 regret bound。也就是：

\[Regret_T(u)=\sum\limits_{t=1}^T \ell_t(\tilde{p}_t)-\ell_t(u)\]

\[\leq R(u)-R(\tilde{p}_1)+\sum\limits_{t=1}^T \ell_t(\tilde{p}_t)-\ell_t(\tilde{p}_{t+1})  -- (5)\]

\[\leq R(u)-R(\tilde{p}_1)+\sum\limits_{t=1}^T \langle \tilde{p}_t-\tilde{p}_{t+1},z_t\rangle , \mbox{ where } z_t\in \partial\ell_t(\tilde{p}_t) -- (6)\]

\[=\frac{1}{2\eta}\|u\|_2^2+\eta\sum\limits_{t=1}^T\|z_t\|_2^2  -- \mbox{ if } R(x)=\frac{1}{2\eta}\|x\|_2^2 -- (7)\]

\[\leq \frac{1}{2\eta}\|u\|_2^2+\eta TL^2  -- \mbox{ if }\ell_t \mbox{ is }L_t-\mbox{Lipschitz} -- (8)\]

### 說明：

(5) 式來自於上面所說：跑 FTL 演算法時當時間 $t$ ，玩 $p_{t+1}$ 其實最好。 
從 (3) 式可以看出，$\tilde{p}_1=arg\min_p R(p)$ 這就像是 FTL 跑在一個「從 $t=0$ 開始的序列。也就是說如果現在 FTL 演算法是跑在某個 loss sequence $\ell_0,\ell_1,\ell_2,\cdots$，那麼 $\tilde{p}_1=p_1,\mbox{ if } R=\ell_0$。所以也可以使用上面的 Lemma，得到

\[R(\tilde{p}_1)+\sum\limits_{t=1}^T \ell_t(\tilde{p}_{t+1})\leq \sum\limits_{t=1}^T \ell_t(u)+R(u),\quad \forall u\in \mathcal{S}\]

左右整理一下，再兩邊同時加上 FTRL 的 total loss $\sum\limits_{t=1}^T\ell_t(\tilde{p}_t)$ 就得到第 (5) 式。

(6) 式從 convex function 的定義得到。
\[\ell_t(\tilde{p}_t)+ \langle \tilde{p}_{t+1}-\tilde{p}_t,z_t\rangle \leq \ell_t(\tilde{p}_{t+1})\]

且事實上 $\partial \ell(p)$ 是 sub-gradient，也就是就算該點不可微分這個不等式也對。而此時問題也從 Online Convex Optimization 簡化成一個 Online Linear Optimization with Regularizer。換句話說，在時間 $t$，

\[\tilde{p}_t=arg\min_p\sum\limits_{\tau=1}^{t-1}\langle z_{\tau},p \rangle+R(p)\]

其中 $\forall \tau, z_{\tau}\in \partial \ell_{\tau}(\tilde{p}_{\tau})$。也就是說在時間 $t$，一旦根據以前的資料預測了 $ \tilde{p}_t$，這個預測的 loss 就被 $\ell_t$ 在 $\tilde{p}_t$ 點的 sub-gradient 決定了。這在程序上是沒有問題的，因為 loss function 本來就可以是看了 $\tilde{p}_t$ 後根據它決定。

(7) 式來自於Online Linear Optimization with $ R(x)=\frac{1}{2\eta}\|x\|_2^2$。
因為

\[\tilde{p}_{t+1}=arg\min_p\sum\limits_{\tau=1}^t \langle z_{\tau},p\rangle+\frac{1}{2\eta}\|p\|_2^2\]

所以很剛好的

\[\tilde{p}_{t+1}=-\eta\sum\limits_{\tau=1}^t z_{\tau}=\tilde{p}_t-\eta z_t\]

因此，$\langle \tilde{p}_t-\tilde{p}_{t+1},z_t\rangle=\eta\|z_t\|_2^2$。

最後，要使得這個 regret bound 是 finite，除了 $u\in \mathcal{S}$ 要是 bounded 以外，$ z_t$ 也都要是 bounded，因此假設 loss function $\ell_t$ is $L_t$-Lipschitz。

這個分析幾乎是將整個 FTRL 的分析模組化了，無論如何，任何 OCO 問題都可以藉由 sub-gradient 簡化成 OLO 問題，而配上不同的 Regularizer 或是不用 $\ell_2$-norm 而是用其他的 norm，那形成的 regret bound 就會不同，不過分析都可以藉由同一個脈絡。

# Reference
1. Online Learning and Online Convex Optimization, by Shai Shalev-Shwartz
