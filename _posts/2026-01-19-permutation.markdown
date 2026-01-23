---
layout: post
title:  "permutation"
date:   2026-01-19 17:21:23 +0900
comments: true
categories: dev
---

### permutation invariance (불변성)
입력에 변화가 생겨도 출력은 변하지 않는 성질

ex. 이미지 분류(Classification)에서 고양이 사진이 왼쪽 아래에 있든 오른쪽 위에 있든, 모델의 최종 출력은 동일하게 '고양이'여야함

### permutation equivariance (등변성)
입력의 변화가 출력에도 동일한 방식으로 반영되는 성질

ex. Segmentation이나 Object Detection에서 입력 이미지에서 고양이가 오른쪽으로 10픽셀 이동하면, 결과물인 마스크나 바운딩 박스도 똑같이 오른쪽으로 10픽셀 이동해야 함

### permutation symmetry
invariance + equivariance