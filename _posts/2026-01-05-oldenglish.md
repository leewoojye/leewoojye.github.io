---
layout: post
title:  "[paper review] AI-Driven Generation of Old English: A Framework for Low-Resource Languages"
date:   2026-01-05 21:02:49 +0900
comments: true
categories: research
---
베이스는 Llama‑8B이고, 파라미터 효율 미세조정(LoRA) + continual pretraining 방식으로 Old English 도메인에 적응, 핵심은 Old English 데이터를 그냥 LM로만 학습하는 게 아니라, 다음 네 가지 “영어-고대영어 연관 태스크”로 학습한다는 점

Text completion (ANG 조각 이어쓰기)

Forward translation (ENG→ANG)

Back translation (ANG→ENG)

Crossed definition (ANG 단어에 대한 ENG 정의)