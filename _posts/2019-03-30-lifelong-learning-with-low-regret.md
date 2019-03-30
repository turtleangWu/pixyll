---
layout: post
title: Lifelong Learning with Low Regret
date: 2019-03-30 12:00
summary: Lifelong learning comes from online learning and multi-task learning. We face tasks and samples sequence by sequence as usual online learning settings. However, there are more than one task which makes learning more difficult.
categories: lifelong
author: Yi-Shan Wu
visible: True
---

# Introduction

Lifelong learning comes from online learning and multi-task learning. We face tasks and samples sequence by sequence as usual online learning settings. However, there are more than one task which makes learning more difficult. How and when can learning sequences of tasks better and easier is what we are interested in. In this paper, and most of previous works about lifelong learning and multi-task learning, tasks are related as they share some common representation, but they are different as each requires a different predictor on top of the representation. We let $\mathcal{G}$ and $\mathcal{H}$ be the space of representation and predictor respectively.
<center class="half">
  <img src="/images/DNN/Gabor_filter.png" width="300" height="250" /><img src="/images/DNN/color_blobs.png" width="300" height="250" />
</center>
