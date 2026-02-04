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