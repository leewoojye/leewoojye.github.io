---
layout: post
title: "existing methods for low-resource language tasks"
date: 2026-02-16 15:37:40 +0900
comments: true
categories: research
---

### Synthetic Data Generation
Scaling Low-Resource MT via Synthetic Data Generation with LLMs (2025)
- 영어 데이터를 기반으로 문서 수준의 합성 코퍼스를 생성하고, 이를 다시 다른 저자원 언어로 피보팅(Pivoting)하여 데이터 부족 문제를 해결

SynOPUS: Scaling Low-Resource MT with LLM-generated Synthetic Data (2025)
- 'SynOPUS'라는 공개 합성 병렬 데이터셋 저장소를 구축

### Cross-lingual & Retrieval based framework
MuRXLS: Multilingual Retrieval-based Cross-lingual Summarization (2025)
- 다국어 검색(Retrieval)을 결합한 인컨텍스트 러닝(In-context Learning) 프레임워크로, 고자원 언어(영어 등)에서 관련 사례를 동적으로 검색하여 저자원 언어의 요약 성능을 도모

Cross-Lingual Transfer Learning for Low-Resource Hate Speech Detection (2025)
- XLM-R과 같은 다국어 사전학습 모델을 고자원 언어에서 먼저 학습시킨 뒤, 최소한의 주석 데이터(Annotated Data)만으로 저자원 언어에 전이 학습(Transfer Learning)을 시키는 최적화 경로를 탐구

Data-Efficient Hate Speech Detection via Cross-Lingual Nearest Neighbor Retrieval with Limited Labeled Data (EMNLP 2025)

AI-Tutor: Interactive Learning of Ancient Knowledge from Low-Resource Languages (ACL 2024)
- 고대 인도어(Prakrit)와 같은 희귀 언어를 대상으로 RAG(검색 증강 생성)와 번역 에이전트를 결합

### Language-Adaptive Fine-Tuning
LoResLM 2025 Workshop (COLING 2025)
- 아프리카 언어(Hausa 등)를 위해 레이블이 없는 대규모 코퍼스로 먼저 적응 학습을 시킨 후 특정 태스크에 맞춰 성능을 극대화

### ours
Neural Proto-language Reconstruction
- 아카드어와 그 후손/형제 언어들의 공통 조상인 **'원시 셈어(Proto-Semitic)'**를 잠재 공간(Latent Space)에서 복원한 뒤 고자원 언어(아랍어)를 조어로 인코딩했다가 다시 아카드어로 디코딩함
- 생성모델, VAE와 유사한 구조
- 생성모델이 자연어처리 도메인에 사용된 사례 조사하기

Phylogenetically-Constrained Contrastive Learning
- 일반적인 mBERT가 모든 언어를 평면적으로 배열한다면, 이 기법은 임베딩 공간 자체를 언어의 진화 계통도와 기하학적으로 일치시킴
- 아랍어로 학습된 지식이 계통상 가장 가까운 아카드어 영역으로 자연스럽게 흘러 들어가는(Knowledge Distillation) 효과를 극대화

Typological Feature Injection & prompt Tuning
- 아카드어의 특성(예: VSO 어순, 교착어적 특성 등)을 벡터화하여 Transformer의 Attention 메커니즘에 Bias로 추가, 텍스트 데이터가 부족하더라도 언어학자들이 이미 밝혀놓은 **'문법적 설계도'**를 모델에게 미리 가르쳐주는 것
- 