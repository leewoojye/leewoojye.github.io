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

요약하자면 학습 흐름은 이렇습니다:

- 데이터 준비: 로봇 시연 데이터를 가져와서 Robot \to→\to→ Human 변환을 수행합니다.
- MLP 학습 (독립적): 위 과정에서 나온 [인간 손 좌표 - 로봇 관절 값] 쌍으로 변환용 MLP를 먼저 학습시켜 둡니다. (이게 준비되어야 나중에 로봇을 실제로 움직일 수 있습니다.)
- VLA 미세 조정: 같은 데이터를 사용하여 **EgoVLA(거대 모델)**를 미세 조정합니다. (영상과 행동의 관계를 배웁니다.)
- 실행(Inference): EgoVLA가 준 MANO 값을 미리 학습해둔 MLP에 넣어서 로봇을 움직입니다.