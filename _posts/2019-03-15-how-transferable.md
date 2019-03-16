---
layout: post
title: How transferable are features in DNN?
date: 2019-03-15 12:00
summary: 大家發現不管是使用哪種 dataset， neural network 在前面幾層學到的東西似乎都很相像、而且型態都偏簡單。這讓我們好奇：是不是前面幾層的學到的資訊其實普遍到是可以共用的？
categories: generalization
author: Yi-Shan Wu
visible: True
---

# 簡介

大家發現不管是使用哪種 dataset， neural network 在前面幾層學到的東西似乎都很相像、而且型態都偏簡單。
<center class="half">
  <img src="/images/DNN/Gabor_filter.png" width="300" height="250" /> 
  <img src="/images/DNN/color_blobs.png" width="300" height="250" />
</center>

這讓我們好奇：是不是前面幾層的學到的資訊其實普遍到是可以共用的？也就是說，如果每個 network 在不同 dataset 學到的東西有某一部份是非常普遍存在的，那麼我是不是能夠重複利用這些資訊，去幫助學習別的 dataset 呢？

重複利用資訊有什麼好處？首先，如果我想學一個新的 dataset ，但那個 dataset 很小以至於 neural network 很容易就 overfit，那這樣是不是就沒望了？不。目前實驗上常見的解法是如果他同樣是個 image classification 問題，那就拿另一個 image 的 dataset 來 pretrain，這功用也相當好像是拿更多額外的資訊來幫助 network 學習，背後基於的精神就是相信 network 在前幾層學到的東西很相像而且可以重複利用。但這畢竟只是目前的假設，是不是真的還需進一步驗證，也是這篇文章要探討的。

# Transfer learning

這種 pretrain 看似只是一個簡單的手段，但是背後的原理也大到自成一門學問，通常大家會稱為 **transfer learning** ，也就是把在 A dataset 學到的資訊轉移（transfer）給 B dataset，而這兩個 dataset 是沒有重複的。Transfer learning 其實也可以視為更 general 的 generalization 問題。在前幾篇文章討論的 generalization 問題中，只希望在 training data 學到的資訊，可以成功給同樣來自於同一個 distribution 的 testing data 使用；而現在的 transfer learning，則是更進一步希望可以用在不一樣的 dataset。一般來說一個成功的 transfer learning 會希望至少要表現的比直接重新學還要好，也就是說，如果重新學 dataset B 可以達到 $77%$ testing accuracy，那麼把 dataset A 的資訊 transfer 到 B 也要至少那麼好。

## 實驗

2014 有人在 ImageNet 上做了這麼一個實驗：把 ImageNet 隨機分成一半當作 dataset A, dataset B，並讓兩個同樣架構的 network 分別學這兩個 dataset，train 在 dataset A 的稱作『 base A』(綠色)而 train 在 dataset B 的稱作『 base B』（紫色）。

<p align="center"><img src="/images/DNN/base_NN.png" width="550" height="180" /></p>

接著下來要複製這兩個 network 的前 n 層 layers 再去學習 dataset B。例如 AnB($^+$) 表示拿學好 A 的前 n 層再去學 dataset B，而 BnB 就是拿學好 B 的前 n 層再去學 dataset B。右上角若有 + 號，表示那被複製的前 n 層的參數能夠再微調，反之則固定不能再動，如下是 A3B($^+$)的圖示。

<p align="center"><img src="/images/DNN/A3B.png" width="450" height="180" /></p>

現在可以來看看實驗結果。

<p align="center"><img src="/images/DNN/transfer_result1.png" width="550" height="350" /></p>

黑色水平虛線是指直接學 B 得到的 testing 結果。A 和 B 是兩組很像的 dataset，畢竟是把 ImageNet 隨機分成一半，因此理想上，我們很希望學習其中一半能夠有效幫助學習另外一半，實驗結果也真的如此，粉紅色的線遠比其他都高。另外，因為 BnB($^+$) 這一組只有 dataset B 的資訊，他的結果如預期只差不多維持原本的樣子。不過這些結果雖然好像很符合我們的直覺，學過 A 的 network 到底擁有什麼樣的資訊使得他能表現的更好？原因還不得而知。注意到 B4B、B5B 的結果竟然比黑色虛線差得多，這蠻令人驚訝的：為什麼固定前 4 或 5 層，只 update 後面幾層沒辦法學的跟原本一樣好？目前作者給的可能的解釋是前後參數的關係是息息相關的，只更新後面而不更新前面比較難找到最佳解。




# 參考資料

1. [How transferable are features in deep neural networks ? (NIPS 2014)](https://arxiv.org/abs/1411.1792)
1. [Convergent Learning : Do different neural networks learn the same representations ? (ICLR 2016)](https://arxiv.org/abs/1511.07543)


