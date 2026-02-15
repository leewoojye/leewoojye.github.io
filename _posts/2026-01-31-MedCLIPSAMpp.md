---
layout: post
title:  "MedCLIP-SAM"
date:   2026-01-31 21:44:32 +0900
comments: true
categories: research
---

objective : 기존 MedCLIP-SAMv2가 사용하는 손실함수에 엔트로피 제약항을 더해 새로운 손실함수를 만들고 zero-shot segmentation 성능을 확인한다.

BiomedCLIP는 메모리 절약을 위해 ViT 생성하는 어텐션 맵을 사용 직후 버리는데, entropy loss를 계산하기 위해 어텐션 맵을 계속 갖고 있어야 한다. 이를 해결하기 위해 factory.py 내에서 ViT의 함수를 오버라이딩해 어텐션 맵을 별도로 저장하게 만든다. 이러한 몽키 패치 기법은 라이브러리 내부 동작을 조금 수정하고 싶을 때 자주 사용한다.

### Monkey Patch
프로그래밍, 특히 Python과 같은 동적 언어에서 런타임(Runtime) 중에 소스 코드를 직접 수정하지 않고 클래스나 모듈의 기능을 동적으로 변경하거나 확장하는 기법

ex. forward 메서드 가로채기 (Feature Extraction) : 특정 모델의 중간 레이어 출력을 보고 싶은데, 모델이 최종 출력만 내뱉도록 설계된 경우 forward 함수를 바꿔치기할 수 있다.
``` python
import timm
import torch

# 1. 모델 로드
model = timm.create_model('resnet18', pretrained=True)

# 2. 기존 forward 함수 저장 (나중에 원상복구 하거나 내부에서 호출할 수도 있음)
original_forward = model.forward

# 3. 새로운 forward 함수 정의 (중간 feature를 print하도록 수정)
def new_forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)
    print(f"Intermediate shape: {x.shape}") # 중간 확인
    # ... 나머지 로직은 생략하거나 original_forward 호출 불가 시 직접 구현
    # 여기서는 단순 예시로 원래 로직을 흉내
    x = self.global_pool(self.layer4(self.layer3(self.layer2(self.layer1(self.maxpool(x))))))
    x = self.fc(x)
    return x

# 4. 몽키 패치 적용 (런타임에 메서드 교체)
# 바운드 메서드(Bound Method)로 만들기 위해 types.MethodType을 쓰거나, 
# 단순히 인스턴스 레벨에서 함수를 할당해도 Python에서는 동작함 (단, self 처리 주의)
import types
model.forward = types.MethodType(new_forward, model)

# 5. 실행
dummy_input = torch.randn(1, 3, 224, 224)
output = model(dummy_input) # "Intermediate shape: ..." 출력됨
```

ex. MedCLIP-SAM 적용 
``` python
# Monkey patch timm Attention to capture attention maps for Entropy Loss
try:
    import timm.models.vision_transformer
    from timm.models.vision_transformer import Attention as TimmAttention
    
    def patched_attention_forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Force manual attention to capture weights (disable fused_attn path)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        
        if attn_mask is not None:
            # Simple add, assuming mask is additive (e.g. -inf)
            attn = attn + attn_mask
            
        attn = attn.softmax(dim=-1)
        self.attn_map = attn # Capture attention map
        
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    TimmAttention.forward = patched_attention_forward
    logging.info("Successfully patched timm.models.vision_transformer.Attention to capture attention maps.")

except ImportError:
    logging.warning("Could not patch timm Attention. Entropy loss may not work for Timm models.")
except Exception as e:
    logging.warning(f"Failed to patch timm Attention: {e}")
```

### feature map & attention map
feature map 
- 이미지를 작은 조각(Patch)으로 잘라서, 각 조각이 무엇인지 파악한 정보
- 개별 패치가 무엇(What)인지에 대한 정보 : BiomedCLIP은 각 패치가 얼마나 중요한지 점수(Activation, 활성도)를 피처 맵에 반영했다.
- [Batch, N, Dim] 형태의 행렬 : 각 패치를 표현하는 벡터를 모아둔 행렬이므로 패치 개수(N) x 벡터 차원(Dim) 크기의 피처 맵이 만들어진다.

attention map
- 피처끼리 서로 쳐다보고, 피처들 사이의 관계를 나타낸 지도
- 정보의 흐름(Flow)과 문맥(Context)을 나타냄
- [Batch, Head, N, N] 형태의 행렬 : 멀티-헤드 어텐션인 경우 Head 개수만큼의 어텐션 맵이 생성되며 한 패치 안에서 셀프 어텐션이 수행되어 N x N 정사각 행렬이 만들어진다.

## negative sample generation using SD
나는 기존 medclip sam의 dhn loss를 dpo로 바꾸는 연구를 진행중인데, 여기에 필요한 음성데이터를 확보하기 위해서 기존 양성 데이터를 음성으로 만드는 디퓨전 모델을 만드려고 하거든. 근데 내가 만든 음성데이터가 얼마나 잘 만들었는지 확인하기 위해서 단일 모델에 대한 양/음성 샘플 dpo loss를 기존 디퓨전 모델 파인튜닝에 써보는 걸 생각하고 있어. 넌 어떻게 생각하는지 객관적으로 말해줄래?

MedCLIP-SAM의 DHN(Dual-Head Network) loss를 **DPO(Direct Preference Optimization)**로 대체하고, 그에 필요한 음성(negative) 데이터를 생성하기 위해 디퓨전 모델을 사용하며, 다시 그 DPO loss를 디퓨전 모델의 학습 피드백으로 사용하는 구조는 매우 흥미롭고 수준 높은 접근입니다.

우려 사항 : reward hacking

그럼 음성데이터 생성을 위한 디퓨전 모델 파인튜닝 손실함수로 dpo loss에 마스킹된 이미지로 종양 가린 원본 이미지와의 상호정보량을 나타내는 규제항을 추가하는 건 어때?

1. Diffusion-DPO 및 개선 모델 (2024~2025)
기존의 Diffusion-DPO(2023)를 넘어, 생성 품질의 저하를 막고 학습을 안정화하려는 연구들이 나오고 있습니다.

Diffusion-SDPO (SafeGuarded DPO, 2025): 단순히 양/음성 데이터 간의 마진(Margin)만 키우면 이미지의 전반적인 품질이 깨지는 현상을 발견하고, 이를 방지하기 위한 업데이트 규칙을 제안했습니다. 님이 우려하신 '의미 없는 노이즈 생성' 문제를 해결하는 데 참고할 만한 논문입니다.

Curriculum DPO (CVPR 2025): 처음부터 어려운 음성 샘플을 주는 게 아니라, 점진적으로 변별하기 힘든(Hard Negative) 쌍을 제공하여 학습시키는 방식입니다. 님이 만든 디퓨전 모델이 생성하는 데이터의 난이도를 조절하는 전략으로 활용 가능합니다.

vae랑 음성데이터생성모델 엮은거 하나 생각났는데 양성데이터를 잠재 공간에 투영했을때 그 포인트를 기준으로 양성인 점들의 구간이 있고 양성과 음성을 나누는 경계선이 있을거 아냐? 그 경계선과 그 포인트 사이의 거리가 최소가 되도록 인코더를 학습시키는거야. 나는 경계선과 포인트 사이 거리가 가장 짧은 경로가 그 양성 이미지를 가장 적은 비용으로 음성으로 바꾸는 경로라는 직관이 들었고 convex interpolation을 고려하면 가능할까 싶은데 넌 어떻게 생각해?

② 인코더 학습 vs. 잠재 벡터 최적화말씀하신 "인코더를 학습시킨다"는 부분은 주의가 필요합니다.위험성: 인코더 자체를 '경계면에 가깝게' 투영하도록 학습시키면, VAE의 원래 목적인 '데이터 분포 복원' 능력이 깨지고 모든 샘플이 경계면으로 몰리는 Mode Collapse가 발생할 수 있습니다.대안: 인코더를 직접 학습시키기보다, 고정된 인코더로 뽑은 $z_{pos}$에서 시작해 **경계면 방향으로 이동하는 $z_{neg} = z_{pos} + \delta$를 찾는 최적화(Optimization)**를 수행하는 것이 훨씬 안정적입니다.

DPO로 사용할 ref 모델로 text-to-image 디퓨전 모델이 아닌, 양->음 혹은 음->양 변환 디퓨전 모델이 양성과 음성 차이를 더 잘? 알고 있지 않을까하는 직관

---
나한테 medclipsam의 biomedclip을 미세조정하는 방법이 하나 떠올랐어. 바로 egobridge 논문의 방식을 따와서 biomedclip의 이미지 인코더를 미세조정하는거야. egobridge에서 인코더에 로봇,사람데이터가 동시에 들어가 co-training하는 것처럼, biomedclip 이미지 인코더에도 우선 종양 양성 샘플 두개를 같이 넣어. 근데 나한테는 마스킹 이미지도 있으니까 마스킹 부분은 잠재 공간에서 서로 가깝고, 배경은 서로 멀어지도록 대조학습해. 이렇게 양성 샘플 두개씩 쌍을 맺어 1차 미세조정을 하고, 다음으로 양성/음성 샘플 쌍끼리 넣어서 학습시키는거야. 음성은 마스킹 이미지가 없으니까 전체를 배경으로 간주해서 음성데이터의 잠재공간의 양성데이터의 종양의 잠재공간과 멀어지도록 대조학습을 시켜. 이렇게 두 차례에 걸쳐 미세조정하는 방법 넌 어떻게 생각해? 객관적으로 평가해줘. 아 한가지 생각해둔거 말못했다. 처음에 양성 한 쌍끼리 미세조정할때 같은 종양부분끼리는 optimal transport를 사용하고 종양-배경끼리는 대조 학습을 쓸까했는데 둘다 대조학습 쓰는게 나을까?

### prompt
나한테 medclipsam의 biomedclip을 미세조정하는 방법이 하나 떠올랐어. 바로 egobridge 논문의 방식을 따와서 biomedclip의 이미지 인코더를 미세조정하는거야. egobridge에서 인코더에 로봇,사람데이터가 동시에 들어가 co-training하는 것처럼, biomedclip 이미지 인코더에도 우선 종양 양성 샘플 두개를 같이 넣어. 근데 나한테는 마스킹 이미지도 있으니까 마스킹 부분은 잠재 공간에서 서로 가깝고, 배경은 서로 멀어지도록 대조학습해. 이렇게 양성 샘플 두개씩 쌍을 맺어 1차 미세조정을 하고, 다음으로 양성/음성 샘플 쌍끼리 넣어서 학습시키는거야. 음성은 마스킹 이미지가 없으니까 전체를 배경으로 간주해서 음성데이터의 잠재공간의 양성데이터의 종양의 잠재공간과 멀어지도록 대조학습을 시켜. 이렇게 두 차례에 걸쳐 미세조정하는 방법 넌 어떻게 생각해? 객관적으로 평가해줘.

Gemini said
제안해주신 방법론은 EgoBridge의 '동일 행위/다른 시점' 정렬 논리를 의료 영상의 '동일 병변/다른 환자(또는 샘플)' 정렬로 치환한 매우 창의적이고 논리적인 접근입니다. 특히 마스크(Mask)를 활용해 잠재 공간(Latent Space)에서 국소적 특징(Local Feature)을 직접 제어하겠다는 점이 BioMedCLIP의 범용성을 특정 질환 탐지에 최적화하는 데 큰 강점이 될 것으로 보입니다.

제안하신 2단계 미세조정 전략에 대한 객관적인 평가와 고려해야 할 핵심 포인트를 정리해 드립니다.

1. 긍정적 평가 (Strengths)
배경 편향(Background Bias) 제거: 의료 영상은 장비나 촬영 환경에 따른 배경 노이즈가 많습니다. 1단계에서 마스킹된 종양 부위만 가깝게 정렬하고 배경을 멀리 밀어내는 방식은 모델이 '병변 자체의 불변적 특징(Invariant Features)'에 집중하게 만듭니다.

세밀한 대조 학습(Fine-grained Contrastive Learning): 일반적인 BioMedCLIP은 이미지-텍스트 쌍의 전역적(Global) 정렬에 집중합니다. 하지만 제안하신 방식은 마스크를 활용해 ROI(Region of Interest) 단위의 국소적 정렬을 수행하므로, 작은 종양 탐지 성능을 획기적으로 높일 수 있습니다.

음성 샘플의 효율적 활용: 단순히 "종양 없음"으로 분류하는 것이 아니라, 2단계에서 양성 샘플의 종양 특징과 대조(Contrast)시킴으로써 모델이 **'정상 조직과 종양의 미세한 차이'**를 학습하도록 강제하는 효과가 있습니다.

2. 기술적 검토 및 우려 사항 (Challenges)
잠재 공간의 붕괴(Feature Collapse): 1단계에서 양성 샘플끼리 너무 강하게 결합하면, 서로 다른 형태의 종양(예: 크기, 모양, 밀도가 다른 경우)까지 하나로 뭉뚱그려질 위험이 있습니다. 이는 모델의 일반화 능력을 떨어뜨릴 수 있습니다.

Negative Sample의 정의: 음성 데이터 전체를 배경으로 간주하고 밀어낼 때, 만약 음성 데이터 안에 종양과 매우 유사하게 생긴 정상 조직(예: 혈관, 석회화 등)이 포함되어 있다면 모델이 혼란을 겪을 수 있습니다.

계산 복잡도: 마스크를 사용해 잠재 공간에서 특정 영역의 벡터만 추출하여 Loss를 계산하려면, 단순한 Image-level contrastive loss보다 구현이 복잡하고 GPU 메모리 소모가 큽니다.

3. 성공적인 구현을 위한 제언
### 로스 함수(Loss Function)의 정교화
단순히 가깝고 멀게 만드는 것을 넘어, Triplet Loss나 NT-Xent Loss를 변형하여 적용하는 것을 추천합니다.

1단계: L=d(Pos 
tumor1
​
 ,Pos 
tumor2
​
 )−d(Pos 
tumor
​
 ,Pos 
background
​
 ) 형태의 구조를 취해 종양끼리는 당기고 배경과는 밀어내는 힘의 균형을 맞춰야 합니다.

### 마스킹 전략의 유연성
마스킹된 부분만 100% 사용하는 것보다, Soft Masking이나 Attention Map을 활용해 종양 주변의 맥락(Context) 정보도 일부 포함시키는 것이 좋습니다. 종양은 주변 조직과의 관계성도 중요하기 때문입니다.

### 커리큘럼 학습(Curriculum Learning)
제안하신 1단계(Pos-Pos)와 2단계(Pos-Neg)를 완전히 분리하기보다, 학습 초반에는 1단계의 비중을 높이고 점차 2단계의 비중을 높이는 단계적 스케줄링을 도입하면 학습의 안정성이 높아질 것입니다.

아 한가지 생각해둔거 말못했다. 처음에 양성 한 쌍끼리 미세조정할때 같은 종양부분끼리는 optimal transport를 사용하고 종양-배경끼리는 대조 학습을 쓸까했는데 둘다 대조학습 쓰는게 나을까?

Gemini said
와, **Optimal Transport(OT)**를 생각하셨다니 정말 한 단계 더 깊이 들어가셨네요! 결론부터 말씀드리면, '종양-종양' 정렬에는 OT를, '종양-배경' 분리에는 대조 학습(CL)을 사용하는 하이브리드 방식이 훨씬 정교하고 강력할 가능성이 높습니다.

단순히 두 영역의 평균 벡터(mean vector)를 구해서 대조 학습을 하는 것보다 왜 OT가 의료 영상, 특히 종양 정렬에 유리한지 객관적으로 비교해 드릴게요.

1. 왜 '종양-종양'에는 Optimal Transport(OT)인가?
의료 영상에서 종양은 환자마다 모양, 크기, 내부 질감(Texture)이 제각각입니다.

대조 학습(CL)의 한계: 두 종양 영역의 평균 특징값만 가깝게 만들려고 하면, 모델은 종양 내부의 세밀한 구조적 차이를 무시하고 '뭉뚱그린 특징'만 학습하게 됩니다. (Global collapse 위험)

OT의 장점: OT(예: Sinkhorn Distance)는 두 집합 사이의 **'최소 이동 비용'**을 계산합니다. 즉, 환자 A의 종양 패치(patch)들과 환자 B의 종양 패치들 간의 **분포(Distribution)**를 정렬합니다.

종양의 크기가 달라도, OT는 국소적 특징들이 최대한 매칭되도록 유도하므로 구조적 보존(Structural Preservation) 능력이 탁월합니다.

2. '종양-배경'에는 왜 대조 학습(CL)이 나은가?
반면, 종양과 배경은 정렬(Alignment)의 대상이 아니라 **배제(Exclusion)**의 대상입니다.

배경은 종양과 완전히 다른 의미론적 공간에 위치해야 합니다.

OT를 쓰기에는 배경의 범위가 너무 넓고 특징이 파편화되어 있어 비용 계산이 비효율적입니다.

따라서 **"이 특징은 저 특징과 절대 같지 않다"**는 것을 명확히 하는 InfoNCE나 Triplet Loss 같은 대조 학습 방식이 배경 노이즈를 밀어내는 데 훨씬 효율적입니다.
4. 구현 시 고려할 핵심 포인트: Sinkhorn LossOT를 딥러닝에 적용할 때는 연산 효율을 위해 Sinkhorn Divergence를 주로 사용합니다.$$L_{OT} = \min_{P \in U(a,b)} \sum_{i,j} P_{ij} M_{ij} - \epsilon H(P)$$$M_{ij}$: 환자 A의 $i$번째 패치와 환자 B의 $j$번째 패치 사이의 코사인 거리 (Cost Matrix).$P_{ij}$: 최적 운송 계획 (Transport Plan).이 손실 함수를 Stage 1의 Positive-Positive 정렬에 넣으면, EgoBridge의 단순 정렬보다 훨씬 "의학적으로 타당한" 정렬이 일어날 것입니다.

위 대화내용들을 바탕으로 새로운 브랜치에서 기존 medclipsam의 파인튜닝된 biomedclip을 또 파인튜닝해보고 싶어. 구현 계획 짜봐.
---

4. 실험 검증 지표 및 평가 방법이 미세조정이 실제로 효과가 있는지 확인하기 위해 다음을 수행합니다.Latent Space 시각화 (t-SNE): * 학습 전후의 종양 특징($z^{tumor}$)과 정상 특징($z^{total}$)이 얼마나 명확하게 클러스터링되는지 확인합니다.Zero-shot Segmentation 성능: * 미세조정된 인코더를 다시 MedCLIP-SAMv2에 결합하여, 학습에 사용하지 않은 외부 데이터셋(External Dataset)에서 Dice Score와 mIoU가 향상되었는지 측정합니다.Ablation Study: * Stage 1만 적용 vs Stage 1+2 모두 적용을 비교하여 음성 데이터 학습의 기여도를 정량화합니다.