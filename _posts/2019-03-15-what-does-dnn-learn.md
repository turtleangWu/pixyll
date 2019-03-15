---
layout: post
title: What does DNN learn?
date: 2019-03-15 12:00
summary: 現在有很多很多人都開始用 deep neural network 協助處理像是分類問題等等，成效也非常好，但是他為什麼可行目前卻沒人曉得，這篇文章主要透過幾篇近年的論文探討這個問題。
categories: generalization
author: Yi-Shan Wu
visible: True
---

# 簡介

以前大家無論是直觀上、還是對一些簡單模型的證明上，都認為當一個模型的可調參數越多，那麼它就越容易 overfit，尤其是當可調參數量遠超過訓練資料量﹝over-parametrized﹞時，overfitting 幾乎難以避免。但是很神奇的是，現在的 neural network 往往超級大，over-parametrized 的狀況幾乎是必然的，不過卻仍然有不錯的普遍化﹝generalization﹞性質。

如果只有部分實驗結果如此，那說不定這種現象只是運氣問題，或者跟初始狀態有關等等，但當大部分的模型訓練出來都有這樣的結果時，會讓我們不禁想：究竟是什麼原因讓機器更樂意去學東西而不是只是背下來呢？機器竟然傾向學習規律，而且現在採用的模型似乎足夠描述這種規律，這是何等其妙的事情。而這個問題在深度學習已經被大量運用在實務上的現在來說，仍然是個大哉問，我們希望盡快地能夠回答這個問題，以幫助在實際建構模型時有更多的依據。

# 名詞解釋

在這些文章中，主要被討論、以及重複提及的概念先在這裡稍微做一下區別，以避免之後的閱讀有一些誤會。

* Generalization（普遍化）：
\[\mbox{Generalization gap}:=L_D(A_S)-L_S(A_S)\]
$S$ 是從 $D$ 這個資料分布選出來的訓練資料，$A$ 是演算法，$A_S$ 是演算法看了訓練資料後選出來的最佳函數，而 $L_S$ 和 $L_D$ 分別代表訓練錯誤率和真實錯誤率。

機器學習的目標是希望能夠估計以及最小化 $L_D(A_S)$ ，而研究 generalization gap 的目的就是希望能夠回答和解釋，什麼樣的情況下，我們對 $L_S(A_S)$ 的估計離 $L_D(A_S)$ 不遠？以往的理論通常希望這個 gap 只會跟演算法 $\mathcal{A}$ 以及 Hypothesis class $\mathcal{H}$ 有關，而跟資料的分佈無關，也就是希望面對任何分佈的資料都能有相同好的預測。

* Overfitting ：
指的是雖然有很低的 training error $L_S(A_S)$（常常幾乎是零），但是 generalization gap 很大。

* Memorization（記憶、背誦）：
指把看過的資料背下來，建立一一對應的關係。舉個例子，當導師第一次踏進一個新接的班級時，在黑板上建立一個表格，把坐在位置上的同學的名字一一紀錄上去，而把那個表和對應的同學的臉記下來就是背誦。

* Learning（學習）：
目前大家理想上希望的『學習』是真實錯誤率可以很低，而這奠基在訓練錯誤率以及 generalization gap 都要很低，只有其中一項很低是不夠的。直觀上，我們希望一個成功的學習，除了可以學會看過的資料（training error 低），而且要能夠舉一反三（generalization gap 低）。

目前大家普遍相信，只會靠『背誦』是沒有辦法成功學習的，因為雖然訓練錯誤率很低，但是不能舉一反三的話 generalization gap 可能很高。而要怎麼樣決定一個 neural network 是否只是『背誦』呢？現在常被採用的方式是餵給機器隨機的資料，而隨機的方式大致上可分成兩種：資料 $ X$ 本身就是一團隨機的資訊、或者是對應的標籤 $Y$ 是隨機給的。後面那個可能稍微好理解一點，也就是例如給你一隻狗用不同角度拍下的 10 張照片，可是跟你說第一張是『狗』，第二張是『貓』，第三張是『老鼠』......等等，此時要將他們全部學起來，大概就只能把他們全部用背的背下來了。

# Randomization tests

今天的故事從一個令人震驚的實驗開始。這些實驗用的是一些實務上能夠成功學習真實資料(true data)的模型。但是下面這個實驗結果卻令人震驚：(a) (b) 圖告訴我們，同一個模型甚至對完全隨機的資料(random data)的訓練錯誤率都能夠達到 0，只要時間夠長。(圖中的 label corruption 是指有多少比例的資料被換成隨機資料。)且 (c) 圖又說，對於真實資料 testing error 很小，但隨著隨機資料的比例越高， testing error 就越大。(因為全部都可被 overfit，所以 testing error 就等於 generalization gap 的大小。)

<img src="/images/DNN/memorize.png" align="center" width="850" height="250" />

實驗使用的 dataset 為 CIFAR-10。(a) 中除了藍線，其他都是經過各種不同 random 方式處理的資料。(c) 中的 testing error 之所以收斂到 0.9 是因為 CIFAR-10 是有 10 個類別的資料。

以前大家普遍相信，*Hypothesis class 的性質*，或是一些 *regularization 的方法*是決定 generalization gap 的因素。但在 2017 年初，這個實驗告訴我們，用以往的方法已經不足以解釋這個現象，許許多多的新方法以及實驗證據被提出，不過直到今日依然沒有個定論。

# Regularization

前面的實驗迫使我們重新思考『機器學習』。機器學習的目標是什麼？首先，對於一個問題，我們會先訂一個『目標函數』﹝objective function﹞，或是 Loss function，而機器的目標就是找出一組參數來最佳化這個目標。那麼機器怎麼決定這組參數好過另一組參數呢？又世上參數有無限多種，我們不可能要求機器大海撈針，找出一個真正最適切的參數，因此我們必須先訂一個 hypothesis class，告訴他在這個範圍裡面搜尋。

但是就算已經訂出一個 hypothesis class，仍然有許多的問題。對於一個很複雜的函數，它可能會有很多的極值，有些極值有好的 generalization gap ，有些沒有。那麼，要怎麼樣告訴機器要選擇哪一個比較好呢？這就牽涉到要如何去『限制』機器的尋找方式，或是在 objective function 動一些手腳，把我們對於『好』的知識加在裡面，使得機器可以避開那些不好的極值，這就是所謂的『Regularization』。

Regularization 主要又可以分為兩大類，一是『Implicit regularization』，另一種是『Explicit regularization』。在實作上，我們常常會在目標函數再加上 $\|\mathbf{w}\|^2$ 的限制，希望選出來的參數可以同時最小化 Loss function 以及 $\|\mathbf{w}\|^2$，代表著我們希望限制函數的複雜度。這就是一種很典型的 Explicit regularization 的方法，告訴機器如果有兩組參數都得到一樣的 Loss，那麼比較不複雜的那個我比較喜歡。

# Implicit regularization

其實上面提到的*hypothesis 的性質*也是一種 Implicit regularization 的方式，它畫了一個範圍告訴機器應該在哪裡尋找答案。而為什麼這目前會被分在『Implicit』呢，筆者目前猜測是因為到底是函數的什麼性質影響了選擇的好壞其實並不是那麼清楚的，尤其是對於 Deep neural network，這樣大的 network 給我們的其實不只是一組參數，而是一堆超級複雜的參數，那究竟是 network 的總參數量影響好壞呢、還是 network 的深度影響好壞、還是其實都不是，這就是近年大家努力探討的問題了。

## 1. Capacity

這個方法是用來估計 hypothesis class 的複雜度的方法，也是決定 generalization gap 的因素之一。最常見的例子像是 VC dimension。

### Norm-based generalization bound

這是另一種用來估計 neural network 複雜度的方法 (Neyshabur, 2015) ，但是這個 bound 同樣太鬆，而且這個 bound 跟資料分佈無關，因此也沒有辦法解釋為什麼實驗結果會跟資料是不是隨機分佈有關。

## 2. Algorithm

另一種被分類在『Implicit regularization』的還有演算法。在實做上，Stochastic Gradient Descent ﹝SGD﹞這個演算法不僅僅是一個比較快的演算法，在近年還意外的發現，使用 SGD 學出來的參數，似乎總是有比較好的 generalization 性質，也就是可以避開那些比較不好的極值。這個發現促使了許多理論的研究，尤其在 2015年後。

### Early stopping

這也是在實作上時常被採用的策略。若是演算法的正確率會爬到某一個高峰後就逐漸下降，那麼採用 early stopping 是有可能可以有比較好的結果的，但是目前為止並沒有一個穩定的證據顯示這個現象。

### SGD

在 2002 年時，一篇『 stability and generalization』的論文中分析了各種不同的 stability 性質。雖然他們有些不同，不過大致上的概念，都是要描述某一種穩定性：『一個演算法當 input 改變一點點時，output 會改變多少』。一個穩定的演算法理想上看到差異很小的兩組資料，那麼他所輸出的函數在那些資料上的行為也不會差太多。在 2002 年的那篇論文中，證明了滿足了『Uniform stability』的演算法，會有不錯的 generalization 性質。而另外在 Hardt 2016 中，作者證明了 SGD 是一個滿足 uniform stability 的演算法，因此 SGD 有不錯的 generalization 性質。

到目前為止看起來都很好。不過很不幸的 uniform stability 的性質也是 data-independent，也就是跟資料無關，因此也沒辦法解釋為什麼會有跟資料的分佈有關係的結果。

# Explicit regularization

除了上述的方法，也可以透過直接加入 regularizer，把一些不可能是好的 hypothesis 剔除掉，而得以縮小那些『有效 hypothesis』的個數。常用的 regularization 像是 weight decay，或是 dropout。Weight decay 其實就等同於對 weights 做 $\ell_2$ regularization；而常用的 dropout 就是在訓練時對每個 neural network 的 node 各別以一定的機率遮住。但以下實驗說明這些方法都不足以解釋。

<img src="/images/DNN/dropout.png" align="center" width="500" height="250" />

由上圖可以發現，對於 original labels，加入 regularizers 後獲得的解的確有更好的 generalization 性質；但是對於 random labels，就算加入 regularizers ，對於 overfitting 的現象也沒有太大的幫助，testing 的結果仍然跟隨意亂猜差不多。也就是說，weight decay、dropout 等等的 regularizer 可能對於 generalization 有一些幫助，但是仍然有許多其他因素是我們未知的。

### 主要參考文章

  1. Understanding deep learning requires rethinking generalization. (Zhang, et al. 2017 ICLR)
  1. Exploring generalization in deep learning (Neyshabur, 2017 NIPS)
  1. A  Closer Look at Memorization in Deep Networks (Arpit, 2017 ICML)
  1. Learning and memorization (Chatterjee, 2018 ICLR)




