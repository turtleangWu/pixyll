---
layout: post
title: Lifelong Learning with Low Regret
date: 2019-03-30 12:00
summary: Lifelong learning comes from online learning and multi-task learning. We face tasks and samples sequence by sequence as usual online learning settings. However, there are more than one task which makes learning more difficult.
categories: lifelong
author: Yi-Shan Wu
visible: False
---

# Introduction

Machine learning algorithms can now solve many problems even better than humans. However, machines are still far from being intelligent that they typically need to relearn when facing new tasks, while humans are able to learn new things efficiently by ultilizing learned knowledge. This motivates the study called lifelong learning (Thrun and Pratt, 1998), which aims to perform better over time by transferring information learned from previously tasks to later ones, under the belief that there are some commonalities across tasks.

<center class="half">
  <img src="/images/lifelong/Lifelong.png" width="800" height="200" />
</center>

In this paper, we face tasks and samples sequence by sequence as usual online learning settings. As introduced above, tasks are related as they share some common representation, but they are different as each requires a different predictor on top of the representation. We let $\mathcal{G}$ and $\mathcal{H}$ be the space of representation and predictor respectively.
