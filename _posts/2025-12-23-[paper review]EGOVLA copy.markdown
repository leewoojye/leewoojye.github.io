---
layout: post
title:  "[paper review] EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos (2)"
date:   2025-12-31 14:09:37 +0900
comments: true
categories: research
---

## EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos 논문 리뷰 (2)

Ego-centric Human Pretraining Improves In-Domain Performance:
Compared to the specialist ACT baselines, generalist models (EgoVLA and EgoVLA-NoPretrain) perform substan- tially better on both short- and long-horizon tasks. This is likely because specialist models must simultaneously learn low-level manipulation and long-horizon planning from scratch, whereas gen- eralist models leverage shared low-level skills across tasks.

Additionally, although EgoVLA is pretrained with a unified action space, it cannot be directly deployed for manipulation without further fine-tuning on a moderate amount of robot data. Future work may explore improving zero-shot transferability through more embodiment-agnostic pretraining