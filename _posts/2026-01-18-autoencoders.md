---
layout: post
title:  "Autoencoders and VAE"
date:   2026-01-18 20:21:16 +0900
comments: true
categories: research
---

![alt text](/assets/images/vae_thumbnail.png)

### 베이즈 정리와 사후확률

사후확률분포 : 어떤 사건이 발생했다는 조건 하에 모델의 파라미터나 가설이 참일 확률

$$P(\theta | D) = \frac{P(D | \theta) P(\theta)}{P(D)}$$

- $P(\theta | D)$ (사후확률, Posterior) : 데이터를 본 후의 파라미터 $\theta$에 대한 확률분포
- $P(D | \theta)$ (우도, Likelihood): 특정 파라미터 $\theta$가 주어졌을 때, 현재 데이터 $D$가 관찰될 확률
- $P(\theta)$ (사전확률, Prior): 데이터를 보기 전, 우리가 $\theta$에 대해 가지고 있던 기존 지식이나 믿음
- $P(D)$ (증거, Evidence): 데이터 자체의 확률로, 보통 정규화 상수 역할

## Variational Inference
### $p(z)$ : 잠재 변수의 사전 분포 (Prior)
항상 표준 정규 분포를 가정하는데 이는 계산을 용이하게 한다.

참고로 가우시안 분포는 평균과 분산이 고정된 분포에 대해 가장 적은 정보를 담은, 즉 불확실성이 가장 큰 분포다. 이는 데이터에 대한 사전 정보가 없을 대 선택하기 좋은 가장 편향되지 않은 분포라고 할 수 있다. 또한 가우시안 분포는 모든 데이터가 0 근처에 적당히 모이게 강제함으로써 규제가 없을 때보다 밀도 높은 분포를 만들 수 있고 이는 새로운 데이터를 생성할 때 연속적이고 매끄러운 생성을 가능하게 한다.

### $q_\phi(z|x)$ : encoder (Approximate Posterior)

### $p_\theta(x|z)$ : decoder (Likelihood)
VAE의 디코더 $p(x|z)$는 잠재 변수 $z$가 주어졌을 때 데이터 $x$가 나올 확률을 모델링하는데, 이때 디코더의 출력 분포를 가우시안 분포로 가정한다.

$$p(x|z) = \mathcal{N}(x; \mu(z), \sigma^2 I)$$

$$\log p(x|z) = \log \left( \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{\|x - \mu(z)\|^2}{2\sigma^2} \right) \right)$$

$$\log p(x|z) = -\frac{1}{2\sigma^2} \|x - \mu(z)\|^2 - \text{constant}$$

여기서 $\|x - \mu(z)\|^2$ 부분이 바로 입력 데이터 $x$와 재구성된 데이터 $\mu(z)$ 사이의 거리를 제곱한 값, 즉 $L_2$ 노름의 제곱이 된다.

일반적인 이미지처럼 연속형 데이터는 가우스 분포를 가정하며, 이진 데이터의 경우 베르누이 분포로 가정하여 이때는 손실 함수가 L2가 아닌 binary cross entropy를 사용한다.

## Evidence Lower Bound
ELBO는 데이터 로그 확률 즉 $\log p(x)$의 하한선이자 VAE에서 학습하는 목적 함수(loss function)다.

$$\log p(x) = \text{ELBO} + KL(q(z|x) \| p(z|x))$$

여기서 KL항이 언제나 0보다 크거나 같으므로 $\log p(x)$는 항상 ELBO보다 크거나 같다.

$$\text{ELBO} = \underbrace{\mathbb{E}_{q(z|x)} [\log p(x|z)]}_{\text{① Reconstruction Term}} - \underbrace{KL(q(z|x) \| p(z))}_{\text{② Regularization Term}}$$

## Reparameterization Trick