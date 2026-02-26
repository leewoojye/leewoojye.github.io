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

biomedclip의 텍스트인코더를 통과한 healthy brain 벡터 빼기 healthy breast 벡터는 tumor brain 벡터 빼기 tumor breast 벡터 결과랑 비슷할까?

1. StyleCLIP (CVPR 2021)CLIP의 텍스트 임베딩 공간이 '선형적'이라는 점을 가장 대중적으로 알린 연구입니다.핵심 아이디어: 특정 속성(예: '안경', '미소')의 텍스트 벡터를 추출한 뒤, 그 차이 벡터를 StyleGAN의 Latent Space($\mathcal{W}$ or $\mathcal{S}$)에 투영하여 이미지를 조작합니다.연관성: 질문하신 $V_{tumor} - V_{healthy}$가 바로 이 연구에서 말하는 **'Global Direction'**에 해당합니다. 이 '질병 방향 벡터'를 일반 장기 영상에 더하면 '질병이 있는 영상'으로 변환할 수 있다는 논리적 근거가 됩니다.

방법 1 (강력 추천): BioMedCLIP 기반 Directional Loss (학습 단계 주입)
가장 확실하게 의미적 차이 벡터를 주입하는 방법은, SD 파인튜닝(LoRA 등) 학습 시의 Loss 체계에 BioMedCLIP을 추가하는 것입니다. (StyleGAN-NADA 등 논문에서 쓰인 CLIP Directional Loss 차용) = Zero-shot Concept Editing

개념: U-Net이 이미지를 인페인팅(수정)하는 **"시각적 변화 방향"**이, BioMedCLIP이 알고 있는 **"언어적 의미 변화 방향"**과 완벽히 일치하도록 강제합니다.
수식:
$\Delta T = V_{text}("healthy\ breast") - V_{text}("tumor\ breast")$ (미리 계산해둔 Text 차이 벡터)
$\Delta I = V_{image}(Generated\ Image) - V_{image}(Original\ Tumor\ Image)$ (진행 중인 이미지 추출 벡터)
$Loss_{dir} = 1 - CosineSimilarity(\Delta I, \ \Delta T)$
구현: SD 모델(U-Net)이 MSE Loss(노이즈 복원)로 유방 이미지를 인페인팅하도록 학습할 때, 보조 Loss(Auxiliary Loss)로 위 $Loss_{dir}$를 추가합니다. 모델은 유방암의 질감을 지우면서 동시에 **"BioMedCLIP이 생각하는 종양 소거의 방향성"**을 U-Net 가중치에 새기게 됩니다. 이렇게 학습된 U-Net은 추후 뇌(Brain) 이미지가 들어와도 텍스처 오버피팅 없이 '종양 소거'라는 고차원적 행위 자체를 수행할 확률이 크게 높아집니다.

[Goal Description]
Implement a Zero-Shot Concept Editing script that uses BioMedCLIP's semantic shift vector ("healthy breast" - "tumor breast") to edit a "tumor brain" image towards a "healthy brain" using Stable Diffusion Inpainting pipeline.