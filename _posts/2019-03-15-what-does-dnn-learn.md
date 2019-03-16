---
layout: post
title: What does DNN learn?
date: 2018-06-21 12:00
summary: 大家發現不管是使用哪種 dataset， neural network 在前面幾層學到的東西似乎都很相像、而且型態都偏簡單。這讓我們好奇：是不是前面幾層的學到的資訊其實普遍到是可以共用的？
categories: generalization
author: Yi-Shan Wu
visible: True
---

# 簡介

大家發現不管是使用哪種 dataset， neural network 在前面幾層學到的東西似乎都很相像、而且型態都偏簡單。這讓我們好奇：是不是前面幾層的學到的資訊其實普遍到是可以共用的？也就是說，如果每個 network 在不同 dataset 學到的東西有某一部份是非常普遍存在的，那麼我是不是能夠重複利用這些資訊，去幫助學習別的 dataset 呢？

重複利用資訊有什麼好處？首先，如果我想學一個新的 dataset ，但那個 dataset 很小以至於 neural network 很容易就 overfit，那這樣是不是就沒望了？不。目前實驗上常見的解法是如果他同樣是個 image classification 問題，那就拿另一個 image 的 dataset 來 pretrain，這功用也相當好像是拿更多額外的資訊來幫助 network 學習，背後基於的精神就是相信 network 在前幾層學到的東西很相像而且可以重複利用。但這畢竟只是目前的假設，是不是真的還需進一步驗證，也是這篇文章要探討的。

# Transfer learning

這種 pretrain 看似只是一個簡單的手段，但是背後的原理也大到自成一門學問，通常大家會稱為 **transfer learning** ，也就是把在 $A$ dataset 學到的資訊轉移（transfer）給 $B$ dataset，而這兩個 dataset 是沒有重複的。Transfer learning 其實也可以視為更 general 的 generalization 問題。在前幾篇文章討論的 generalization 問題中，只希望在 training data 學到的資訊，可以成功給同樣來自於同一個 distribution 的 testing data 使用；而現在的 transfer learning，則是更進一步希望可以用在不一樣的 dataset。





# 參考資料

1. [How transferable are features in deep neural networks ? (NIPS 2014)](https://arxiv.org/abs/1411.1792)
1. [Convergent Learning : Do different neural networks learn the same representations ? (ICLR 2016)](https://arxiv.org/abs/1511.07543)


