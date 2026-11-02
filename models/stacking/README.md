# Stacking (최종 선정 · Stacking B) · 한승아님

# Stacking 모델 학습 및 성능 분석

## 1. 모델 개요

호텔 예약 취소 여부를 예측하기 위해 서로 다른 특성을 가진 모델들을 결합하는 **Stacking Ensemble** 기법을 적용하였다.

단일 모델의 예측 결과를 결합하여 예측 성능과 일반화 성능을 개선하고자 하였으며, 두 가지 Stacking 구조를 비교하였다.

| 구분 | Base Models | Meta Model |
|---|---|---|
| Stacking A | XGBoost + Random Forest + Logistic Regression | Base Model의 예측 확률을 활용한 최종 분류 모델 |
| **Stacking B** | **XGBoost + Random Forest + MLP** | Base Model의 예측 확률을 활용한 최종 분류 모델 |

최종적으로 Stacking B를 선정하였다.

## 2. 모델 학습 구조

### 2.1 Stacking 학습 과정

1. 공통 전처리 데이터로 Train / Validation / Test 데이터셋 구성
2. `StratifiedGroupKFold(n_splits=5)`를 활용하여 그룹 기반 교차검증 수행
3. 각 Base Model의 OOF(Out-of-Fold) 예측 확률 생성
4. OOF 예측 확률을 새로운 입력 특성으로 활용하여 Meta Model 학습
5. 학습 완료된 Base Model의 예측 확률을 Meta Model에 전달하여 최종 취소 확률 산출
6. Validation 및 Test 데이터로 예측 성능 평가

**OOF 예측을 활용한 이유**

Base Model이 학습에 사용했던 데이터에 직접 예측한 값을 Meta Model 학습에 사용하면 과적합이 발생할 수 있다. 따라서 각 Fold에서 학습에 사용하지 않은 데이터의 예측값을 생성하여 Meta Model의 학습에 활용하였다.

또한 동일한 예약 조건을 가진 데이터가 서로 다른 Fold에 분포하지 않도록 그룹 기반 교차검증을 적용하였다.

## 3. 전처리 및 특성 최적화

### 3.1 One-Hot Encoding 희소 범주 처리

`agent`, `country`와 같이 범주 수가 많은 변수는 One-Hot Encoding 시 차원이 크게 증가할 수 있어 `min_frequency`를 조정하는 실험을 진행하였다.

| 변수 | 전체 범주 수 | 20회 미만 | 100회 미만 |
|---|---:|---:|---:|
| agent | 300개 | 143개 (0.98%) | 229개 (5.50%) |
| country | 166개 | 98개 (0.63%) | 134개 (2.73%) |

괄호 안의 비율은 해당 저빈도 범주에 속하는 데이터가 전체 행에서 차지하는 비율이다.

`min_frequency=20`과 `100`을 비교하여 희소 범주 통합에 따른 모델 복잡도와 예측 성능 변화를 확인하였다.

### 3.2 특성 중요도 분석

Permutation Importance를 활용하여 개별 특성이 최종 Stacking 모델의 ROC-AUC에 미치는 영향을 분석하였다.

주요 특성으로는 `country`, `deposit_type`, `agent`, `lead_time`, `total_of_special_requests` 등이 나타났다.

반면 중요도가 매우 낮거나 음수에 가까운 일부 특성을 확인하여 제거 실험을 수행하였다.

**제거 대상으로 선정한 특성**

- `has_company`
- `stays_in_weekend_nights`
- `arrival_date_year`

이 중 `arrival_date_year`는 실제 서비스에서 연도 입력에 의존하지 않도록 하기 위한 목적도 포함된다.

파생변수인 `total_guests`, `total_nights`는 생성 후 원본 특성을 제거하는 방식으로 처리하였다.

## 4. 하이퍼파라미터 튜닝 및 과적합 완화

### 4.1 Base Model별 튜닝

| 모델 | 주요 조정 항목 | 목적 |
|---|---|---|
| XGBoost | `max_depth`, `min_child_weight`, L2 규제 | 트리 복잡도 및 가중치 크기 제어 |
| Random Forest | `max_depth`, `min_samples_leaf` | 과도한 분할 방지 |
| MLP | `alpha`, `early_stopping`, `max_iter` | 가중치 규제 및 불필요한 반복 학습 방지 |

### 4.2 MLP 학습 설정

| 파라미터 | 설정 |
|---|---|
| `max_iter` | 300 |
| `early_stopping` | True |
| `n_iter_no_change` | 15 |
| `alpha` | 0.01 |

MLP는 최대 300회 반복 학습하도록 설정하였으며, Early Stopping을 통해 검증 성능 개선이 일정 기간 나타나지 않으면 학습을 종료하도록 하였다.

또한 L2 가중치 패널티를 적용하여 모델의 복잡도를 제어하였다.

## 5. 최종 모델 성능 평가

### 5.1 Stacking B 평가 결과

`min_frequency=100` 설정에서 확인한 평가 결과는 다음과 같다.

<img width="700" height="227" alt="image" src="https://github.com/user-attachments/assets/147f1f62-a396-4e40-8cd4-f2e1958d26d1" />


| 평가 지표 | Train | Validation | Test |
|---|---:|---:|---:|
| Accuracy | 0.8620 | 0.8611 | **0.8658** |
| F1-score | 0.8071 | 0.8042 | **0.8108** |
| ROC-AUC | 0.9364 | 0.9397 | **0.9402** |
| PR-AUC | 0.9097 | 0.9139 | **0.9145** |
| Brier Score ↓ | 0.0987 | 0.0970 | **0.0958** |

### 5.2 평가 지표 해석

- **Accuracy:** Test 정확도 0.8658로 전체 예약의 약 86.6%를 올바르게 분류하였다.
- **F1-score:** 0.8108로 Precision과 Recall 사이의 균형을 확인하였다.
- **ROC-AUC:** 0.9402로 예약 취소와 유지 고객을 구분하는 능력이 우수하게 나타났다.
- **PR-AUC:** 0.9145로 취소 고객을 중심으로 한 Precision-Recall 성능을 확인하였다.
- **Brier Score:** 0.0958로 취소 확률 예측의 평균제곱오차가 낮게 나타났다.

## 6. 과적합(Overfitting) 검증

### 6.1 데이터셋별 성능 비교

학습용(Train), 검증용(Validation), 평가용(Test) 데이터의 성능을 비교하여 모델의 일반화 성능과 과적합 가능성을 점검하였다.

- **ROC-AUC:** Train 0.9364, Validation 0.9397, Test 0.9402로 데이터셋 간 큰 차이가 나타나지 않았다.
- **F1-score:** Validation 0.8042, Test 0.8108로 평가용 데이터에서도 성능이 유지되었다.
- **Brier Score:** Validation 0.0970에서 Test 0.0958로 감소하여 예측 확률의 오차가 소폭 줄어들었다.

### 6.2 과적합 판단

Train, Validation, Test의 ROC-AUC 차이가 작고 평가용 데이터에서 급격한 성능 저하가 관찰되지 않았으므로, **뚜렷한 과적합 징후 없이 비교적 안정적인 일반화 성능을 확보한 것으로 판단하였다.**

다만 OOF 예측값을 활용해 학습한 Meta Model의 Train 성능은 완전히 독립적인 평가 결과가 아니므로, 해당 수치만으로 과적합이 없다고 단정할 수는 없다.

### 6.3 과적합 완화를 위한 적용 방법

- `StratifiedGroupKFold` 5-Fold 교차검증을 통한 그룹 중복 방지
- Base Model의 OOF 예측 확률을 활용한 Meta Model 학습
- Pipeline 내부에서 Fold별 전처리 학습
- XGBoost, Random Forest의 복잡도 제어
- MLP의 L2 규제 및 Early Stopping 적용
- Validation 데이터 기반 성능 비교

## 7. 최종 결론

Stacking B는 XGBoost, Random Forest, MLP의 예측 결과를 결합하여 호텔 예약 취소 여부를 예측하는 모델이다.

Test 데이터에서 **ROC-AUC 0.9402, F1-score 0.8108**을 기록하였으며, Validation과 Test의 성능 차이가 작아 안정적인 예측 결과를 보였다.

또한 특성 중요도 분석과 전처리 최적화 실험을 통해 모델 성능뿐 아니라 실제 서비스에서의 입력 편의성과 활용 가능성도 고려하였다.

향후에는 새로운 예약 데이터에 대한 성능 검증과 확률 보정 등을 통해 모델의 실무 활용도를 높일 수 있을 것으로 기대한다.
