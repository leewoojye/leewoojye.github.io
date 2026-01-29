---
layout: post
title:  "XAI technique, gScoreCAM"
date:   2025-12-29 15:50:21 +0900
comments: true
categories: research
---

![alt text](/assets/images/image-19.png)

MedCLIP-SAMv2에서 information bottleneck M2IB은 이미지–텍스트 둘 다 잘 맞추면서, 이미지 쪽에서 진짜 필요한 패치 정보만 통과시키도록 강제해 saliency map을 만드는 역할을 한다.

M2I는 텍스트 프롬프트 T T와 관련된 부분만 남기고, 나머지 시각 정보는 최대한 버려서 saliency map(이미지에서 어느 부분이 프롬프트와 연관이 높은지 나타내는 맵) $λs$ 를 얻는 것을 목표로 한다.

따라서 M2I 목적함수는 
- $I(Z,Z_{text})$: 맵과 텍스트 사이 mutual information은 키우고, ($Z$는 $Z_{img}$에 mask를 곱한 saliency map 그 자체)
- $I(Z,I)$: 맵과 원본 이미지 사이 mutual information은 줄이는 방향으로 최적화한다. (몇몇 패치만 봐서는 원본 이미지 전체를 reconstruct 못 하게, 즉 정보를 최소한만 남기기 위한 제약)
- 따라서 M2IB가 학습하는 마스크 $λs$ 는 다음과 같다.
$$\lambda_S = \arg\max_{\lambda} \text{I}(Z, Z_{\text{text}}) - \gamma \text{I}(Z, I)$$

$\gamma$ (라그랑주 승수): 압축의 정도와 정보 유지 사이의 균형을 조절하는 하이퍼파라미터

### M2I 파이프라인
- CLIP의 텍스트, 비전 인코더를 거친 토큰들은 각각 CLS 임베딩인 $Z_{text}$, $Z_{img}$에 집약되는데, 이때 $Z_{text}$ 안에는 “tumor, lesion, nodule” 같이 진단에 중요한 단어들의 정보가 더 크게 반영되고, 배경 단어는 상대적으로 덜 반영되도록 학습되어 있는 상태다.
  - 즉! M2I 입장에서는 종양이라는 단어만 따로 보는게 아니라, 종양이라는 개념이 강하게 encode된 문장 벡터 $Z_{text}$를 받는 것에 가깝다.
- $Z_{img}$ 위에 $λs ∈[0,1]^{H×W}$ 형태의 stochastic mask를 곱해서 정보를 통과시킬지 말지를 픽셀/패치 단위로 조절한다.
- $λs$ 를 통해 마스킹된 이미지 표현이 텍스트와는 충분한 mutual information을 유지하면서, 원본 이미지와의 불필요한 정보는 줄이도록 하는 M2I objective를 최적화한다.

참고로 M2I는 이 둘을 받아서 stochastic mask $λs ∈[0,1]^{H×W}$ 를 출력하는 작은 네트워크(파라미터를 가진)로 구현돼 있다고 한다.