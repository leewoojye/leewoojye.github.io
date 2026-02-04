---
layout: post
title:  "자주 사용하는 리눅스 명령어"
date:   2026-01-04 15:39:24 +0900
comments: true
categories: dev
---

ps aux | grep "python inference_test.py"

nohup sh -c 'PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 sh scripts/finetune_chartgemma_4b.sh' > finetune_gemma.log 2>&1 &

scp 로컬디렉토리주소 사용자명@서버IP주소:원격디렉토리주소

ssh -p 9404 사용자명@서버IP주소

pip install git+https://github.com/mattloper/chumpy --no-build-isolation

![alt text](/assets/images/image-1.png)

Kaggle API를 사용하여 특정 파일만 받기

kaggle datasets download -d [계정명]/[데이터셋명] -f [파일명] (예: kaggle datasets download -d zynicide/wine-reviews -f winemag-data-130k-v2.csv)

번외로 kaggle API 활용하는 방법 이외에 Kaggle 노트북을 활용해 샘플을 추출하는 방법이 있습니다. 각 데이터셋 페이지마다 notebook을 생성해 데이터셋을 실험해볼 수 있습니다.

``` python
import pandas as pd
# 데이터 불러오기 (경로는 오른쪽 패널에서 복사 가능)
df = pd.read_csv('/kaggle/input/dataset-name/file.csv', nrows=100) 
# 샘플만 저장하기
df.to_csv('sample_data.csv', index=False)
```