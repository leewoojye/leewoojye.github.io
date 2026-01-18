---
layout: post
title:  "Shallow transfer learning for NLP (1)"
date:   2026-01-18 18:28:12 +0900
comments: true
categories: dev
---

The goal of domain adaptation is to modify, or adapt, data in a different target domain in such a way that the pretrained knowledge from the source domain can aid learning in the target domain. We apply a simple autoencoding approach to “project” samples in the target domain into the source domain feature space.

![alt text](<Screenshot 2026-01-18 at 6.44.02 PM.png>)

FastText is known for its ability to handle out-of-vocabulary words, which comes from it having been designed to embed subword character n-grams, or subwords (versus entire words, as is the case with word2vec). This enables it to build up embeddings for previously unseen words by aggregating composing character n-gram embeddings.

즉 word2vec은 학습 시점에 보지 못한 단어(OOV)에 대해서는 별도 벡터를 만들 수 없지만, fastText는 subword(문자 n-gram) 기반이라 새로운 단어도 그 안에 포함된 문자 n-gram 임베딩들을 합쳐 벡터를 만들 수 있다.

<!-- n-gram은 텍스트나 발화에서 연속된 n n개의 단어(또는 문자 등)를 하나의 단위로 보는 개념 -->

### Multitask learning
Traditionally, machine learning algorithms have been trained to perform a single task at a time, with the data collected and trained on being independent for each separate task. 
This is somewhat antithetical to the way humans and other animals learn, where training for multiple tasks occurs simultaneously, and information from training on one task may inform and accelerate the learning of other tasks.
This additional information may improve performance not just on the current tasks being trained on but also on future tasks, and sometimes even in cases where no labeled data is available on such future tasks. This scenario of transfer learning with no labeled data in the target domain is often referred to as **zero-shot transfer learning**.