---
layout: post
title: Short Introduction to Lifelong Learning
date: 2019-08-29 12:00
summary: Lifelong Machine Learning (LML) is a paradigm to make the machine reach close to the human level of intelligence.
categories: Lifelong-Learning
author: Yi-Shan Wu
visible: True
---

Lifelong Machine Learning (LML) is a learning paradigm to make the machine reach close to the human level of intelligence. The characteristics of LML is to learn continuously over time by leveraging learned knowledge and transferring to unseen tasks. The motivation comes from the recognition that although the current machine learning (ML) paradigm has been able to outperform human beings in several tasks, there are still obvious shortcomings. 

First, current ML methods learn tasks separately. That is, when facing new tasks, we have to collect corresponding data and train a new model, which requires a great amount of data to effectively learn a new task. Furthermore, current theoretical results are mainly based on statistics, which only guarantees generalization on testing data with the same distribution as training data. Although there are now researches on transfer learning and multi-task learning that can deal with more than one task, they can only treat a few tasks and hard to learn over time. To be specific, multi-task learning only targets $$N$$ chosen tasks and try to learn well on them by sharing common knowledge among tasks. The shared knowledge leads to requiring fewer examples to learn all these tasks well. The more knowledge they shared, the fewer examples they required. Transfer learning goes one step further that it aims to perform well on an unseen task, but this is still far from learning continuously over time for future tasks.

In the past few years, a line of work known as continual learning popped out in the deep learning community. The problem aims to learn continuously over time even adapt to new tasks. However, there are problems of catastrophic forgetting that the algorithms tend to forget previous knowledge when learning over time. In contrast, humans can accumulate learned knowledge and used them for future tasks that we often require a few new examples to get familiar with new tasks.

