---
layout: post
title: What does DNN learn?
date: 2019-03-15 12:00
summary: 現在有很多很多人都開始用 deep neural network 協助處理像是分類問題等等，成效也非常好，但是他為什麼可行目前卻沒人曉得，這篇文章主要透過幾篇近年的論文探討這個問題。
categories: generalization
author: Yi-Shan Wu
visible: True
---

### 簡介

以前大家無論是直觀上、還是對一些簡單模型的證明上，都認為當一個模型的可調參數越多，那麼它就越容易 overfit，尤其是當可調參數量遠超過訓練資料量﹝over-parametrized﹞時，overfitting 幾乎難以避免。但是很神奇的是，現在的 neural network 往往超級大，over-parametrized 的狀況幾乎是必然的，不過卻仍然有不錯的普遍化﹝generalization﹞性質。

如果只有部分實驗結果如此，那說不定這種現象只是運氣問題，或者跟初始狀態有關等等，但當大部分的模型訓練出來都有這樣的結果時，會讓我們不禁想：究竟是什麼原因讓機器更樂意去學東西而不是只是背下來呢？機器竟然傾向學習規律，而且現在採用的模型似乎足夠描述這種規律，這是何等其妙的事情。而這個問題在深度學習已經被大量運用在實務上的現在來說，仍然是個大哉問，我們希望盡快地能夠回答這個問題，以幫助在實際建構模型時有更多的依據。

### 名詞解釋

在這些文章中，主要被討論、以及重複提及的概念先在這裡稍微做一下區別，以避免之後的閱讀有一些誤會。

* Generalization（普遍化）：
\[\mbox{Generalization gap}:=L_D(A_S)-L_S(A_S)\]
$S$ 是從 $D$ 這個資料分布選出來的訓練資料，$A$ 是演算法，$A_S$ 是演算法看了訓練資料後選出來的最佳函數，而 $L_S$ 和 $L_D$ 分別代表訓練錯誤率和真實錯誤率。

機器學習的目標是希望能夠估計以及最小化 $L_D(A_S)$ ，而研究 generalization gap 的目的就是希望能夠回答和解釋，什麼樣的情況下，最小化 $L_S(A_S)$ 是一個有效降低 $L_D(A_S)$ 的方法。以往我們通常希望這個 gap 只會跟演算法 $\mathcal{A}$ 以及 Hypothesis class $\mathcal{H}$ 有關，而跟資料的分佈無關，也就是希望面對任何分佈的資料都能有相同好的預測。

* Overfitting ：
指的是雖然有很低的 training error $latex L_S(A_S)$﹝常常幾乎是零﹞，但是 generalization gap 很大。

* Memorization（記憶、背誦）：
指把看過的資料背下來，建立一一對應的關係。舉個例子，當導師第一次踏進一個新接的班級時，在黑板上建立一個表格，把坐在位置上的同學的名字一一紀錄上去，而把那個表和對應的同學的臉記下來就是背誦。

* Learning（學習）：
目前大家理想上希望的『學習』是真實錯誤率可以很低，而這奠基在訓練錯誤率以及 generalization gap 都要很低，只有其中一項很低是不夠的。直觀上，我們希望一個成功的學習，除了可以學會看過的資料﹝training error 低﹞，而且要能夠舉一反三﹝generalization gap 低﹞。
