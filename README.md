# 항만 입항 기록 ↔ 선박 카탈로그 매칭 (DACON 항만 문제해결 AI 공모전)

입항 기록 한 행(선박명·호출부호·톤수 2종·국적·선종·입항 연도)을 공식 선박 카탈로그 78,595척 중 하나의 `vessel_id`에 연결하는 문제입니다.

- 평가: `0.7 × 한글 선박명 행 정확도 + 0.3 × 비한글 행 정확도`
- 제약: 코드 제출(L4 GPU, 추론 30분), **행 독립 예측**(다른 test 행·전체 분포 사용 금지), 외부 선박 데이터 금지, train.csv와 vessel_catalog.csv만 학습에 사용
- 현재 최고 Public: **0.9356 (v42b)** — 한글 489/532, 비한글 3,475/3,568 (점수 소수 10자리로 역산)

저장소에는 **대회 데이터(train.csv, vessel_catalog.csv)가 없습니다.** DACON에서 받아 `data/open/`에 두면 됩니다.

## 점수 변화 (Public)

| 버전 | 핵심 변경 | Public | 한글 맞힌 수 (532) | 비한글 (3,568) |
|---|---|---:|---:|---:|
| v31 | 영문 퍼지/부분 일치 + 한글 학습 음차(EM+PMI) + 로마자 + 톤수 창 검색, LightGBM lambdarank (89 특징) | 0.9125 | 472 | 3,466 |
| v33 | + IDF 토큰 특징 5개 (94 특징) | 0.9183 | 477 | 3,457 |
| e8 | v31·v33 계열 시드 앙상블 8개 (행별 z-점수 평균) | 0.9219 | 479 | 3,469 |
| v38 | 한글은 e8, 비한글은 12개 앙상블 | 0.9224 | 479 | 3,475 |
| v42 | + 한글 퍼지 후보 검색 (오타 되돌림 포함, K=200) | 0.9303 | 485 | 3,475 |
| **v42b** | 퍼지 후보 K=500 | **0.9356** | **489** | 3,475 |
| v44 / v44p | 한글 랭커 재학습: 퍼지 후보 목록 + 주최 측 방식 음절 오타 증강 (11/01 제출, 결과 대기) | – | – | 3,475 |

홀드아웃(fold 0 배를 학습에서 제외, 주최 측 방식 오타 386행): 기존 0.9145 → v42b 0.9404 → v44 0.9456(새 랭커만) / 0.943(기존과 혼합). `logs/noise_f0_*.log`.

효과가 없었거나 손해였던 것: train에서 본 이름 기억(v34/v35, −0.013), 톤수 우도 특징(v37), 규칙 기반 1등 교체(CV에서 고친 것보다 망친 게 10배), 2단계 duel 모델, xendcg·이진 분류·큰 트리 랭커(CV 손해 또는 중립), 비한글 전용/한글 전용 랭커(중립), "한국 입항 배" 사전확률(중립).

## 접근 방법

```
입항 기록 1행 (행마다 독립)
  │
  ├─ 후보 검색  (pipe.py ~ pipe23.py)
  │    영문: 압축 이름 fuzz.ratio / partial_ratio, 토큰 집합, 이니셜, 톤수 창(±6%) + 선종·국적 구조 점수
  │    한글: 자음 골격·모음 키 편집 거리, 학습 음차기(EM으로 train 쌍에 맞춘 transducer, PMI 점수,
  │          단어 생략 허용 local 점수), 로마자 변환, 표준 표기 토큰 사전(tok_ko.json), 톤수 창 채널
  │          + v42: 퍼지 채널(골격/로마자/PMI/부분 골격 상위 K) + 오타 되돌림(인접 음절 교환·중복 음절 제거)
  │    train.csv에 나온 배는 후보에는 두고 최종 선택에서만 제외 (평가 배는 train과 겹치지 않음)
  │
  ├─ 특징 94개  문자열 유사도, 골격, 숫자 일치, 톤수 차·정확 일치, 선종/국적 대응 확률, 순위·격차,
  │             음차 PMI·단위당 점수·2등과의 차, 토큰 매칭, IDF
  │
  └─ 순위 학습  LightGBM lambdarank (lr 0.05, 63 leaves, 500 rounds), 선박 단위 5-fold
               학습 데이터: train 행 + 증강(톤수 삭제, 오타, 약칭, 접두어 교체, 절단, 어순, 개명) + 카탈로그에서
               합성한 한글 질의(synth5.py). 앙상블은 랭커 점수를 행 안에서 z-정규화해 평균.
```

진단 제출(한글 행만 예측 / 2등 후보 예측 / 상위 3개 밖 후보 예측)로 Public 오답을 "2등에 밀린 것"과 "후보에 없는 것"으로 나눠 봤습니다(`tpl_probe*`). 점수가 소수 10자리라 모든 제출에서 한글·비한글 맞힌 수를 정확히 역산할 수 있습니다(`src/lb_decode.py`, `src/probe_decode.py`).

## 저장소 구조

```
src/            파이프라인(pipe*.py: v1→v23 상속 구조, 최신은 pipe23.py), 음차(translit.py), 로마자(rom.py),
                이름 정규화(vm.py), 증강(aug.py, aug2.py, synth.py, synth5.py), 학습 데이터 생성(build_feats*.py),
                학습·CV(cv_eval_mm*.py, train_*.py), 분석 스크립트(*_cv.py, noise_diag.py, lb_decode.py ...)
scripts/        제출 패키징(pack_scoped.sh, pack_ensemble.sh: 시뮬레이션 + 행 독립 검사 + zip), 노이즈 평가(run_noise*.sh)
submissions/    제출 패키지의 script.py·requirements.txt·model/*.py·특징 목록. submit_v42b_fuzzy500에는 랭커 12개 포함
                (train.csv는 제외: 각 패키지 model/ 에 train.csv를 복사해야 실행됨)
rankers/        v44 랭커 3개(시드 42/7/2026, 전체 fold) + fold-0 홀드아웃 랭커
cvres/          fold별 OOF 결과 csv
logs/           CV·진단 로그 (오답 순위 분포, 규칙 CV, 노이즈 세트 결과, 오답별 진단)
```

## 재현

```bash
pip install -r submissions/submit_v42b_fuzzy500/requirements.txt
# 제출 패키지 실행: data/test.csv, data/vessel_catalog.csv 를 두고
cp data/open/train.csv submissions/submit_v42b_fuzzy500/model/
cd submissions/submit_v42b_fuzzy500 && python script.py        # output/submission.csv

# 학습 데이터 생성 → 랭커 학습 (fold 단위, 메모리 6GB 기준)
python src/build_feats26.py v44 1,2,3,4,0 600 3000             # 한글 전용, 퍼지 후보 + 주최 측 방식 오타
python src/train_v44.py mat && python src/train_v44.py fit ranker_v44.txt 42
```

## 규칙 준수

- 학습에는 train.csv와 vessel_catalog.csv만 썼습니다. 외부 선박 데이터·매핑 테이블은 쓰지 않았습니다.
- 모든 예측은 행 독립입니다. 패키징 때마다 3,000행 시뮬레이션에서 행 섞기·1행·부분집합 실행 결과가 같은지 확인합니다.
- call_sign으로 행끼리 묶거나 train을 조회하는 기능은 넣지 않았습니다.
