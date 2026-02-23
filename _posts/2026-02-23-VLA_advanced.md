---
layout: post
title: "논문 찾는 방법 정리"
date: 2026-02-23 21:06:50 +0900
comments: true
categories: research
---

### UniTacHand: Unified Spatio-Tactile Representation for Human to Robotic Hand Skill Transfer
인간의 햅틱 장갑(haptic glove)으로 수집한 촉각 데이터를 로봇 손(dexterous robotic hand)으로 효과적으로 전이(transfer)하기 위한 통합 표현 방식을 제안

2. 표현 학습: 대조 학습을 통한 잠재 공간 정렬단순히 UV 맵으로 투영하는 것만으로는 충분하지 않기에, 연구팀은 두 도메인의 데이터를 하나의 공통된 잠재 공간(latent space)으로 정렬하는 \(2\)단계 학습 과정을 거칩니다.Paired Data 활용: 약 \(10\)분 분량의 짧은 인간-로봇 동시 조작 데이터를 사용하여 **Contrastive Learning(대조 학습)**을 수행합니다.

손실 함수: 두 도메인의 임베딩을 가깝게 만드는 InfoNCE Loss (\(L_{CON}\))

공통 잠재 공간에서 원래의 촉각 정보를 복원해내는 Reconstruction Loss 

(\(L_{REC}\))도메인 간의 차이를 지우기 위한 Adversarial Loss (\(L_{ADV}\))

### METIS: Multi-Source Egocentric Training for Integrated Dexterous Vision-Language-Action Model

2. 효율적인 표현 학습: Motion-aware Dynamics고차원의 연속적인 손 동작을 모델링하는 것은 LLM 기반 VLA 모델에게 매우 도전적인 과제입니다. METIS는 이를 이산화된 토큰(Discretized Tokens) 형태로 변환하여 해결합니다.시각적 동역학(Visual Dynamics): \(VQ-VAE\)를 사용하여 손과 물체 사이의 상호작용으로 인한 시각적 변화를 압축적으로 인코딩합니다.모션 동역학(Motion Dynamics): \(RQ-VAE(Residual Quantization VAE)\)를 사용하여 미세한 손가락 움직임을 계층적으로 양자화합니다.이러한 방식은 단순한 액션 빈(Action Bin) 방식보다 **미세한 조작(Fine-grained motion)**을 훨씬 더 잘 포착하게 해줍니다.