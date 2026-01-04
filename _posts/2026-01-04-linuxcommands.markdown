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