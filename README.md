# SKN37-2ST-3TEAM

# 🏨 고객 이탈 예측 서비스 (Customer Churn Prediction App)

> **호텔 예약 취소 및 고객 이탈 가능성을 예측하고, 데이터 기반의 리텐션 전략을 제안하는 Streamlit 웹 애플리케이션 서비스입니다.**

<br/>

## 🛠 기술 스택 (Tech Stack)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F79939?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)

<br/>

---

## 📌 목차 (Table of Contents)
1. [프로젝트 설계](#1-프로젝트-설계)
2. [데이터 전처리 및 EDA](#2-데이터-전처리-및-eda)
3. [모델링 및 결과 해석](#3-모델링)
4. [Streamlit 앱 개발](#4-streamlit-앱-개발)

<br/>

---

## 1. 프로젝트 설계

### 🎯 비즈니스 목표 및 문제 정의
- **문제 배경**: 호텔 업계에서 예약 취소(이탈)는 객실 점유율 저하 및 직간접적인 매출 손실을 유발합니다.
- **프로젝트 목표**: 
  1. 예측 시점을 **예약이 확정된 직후**로 설정하고, 이때 알 수 없는 정보(최종 상태, 예약 변경 등)를 배제하여 실무 적용 가능한 정밀 이탈 예측 모델 구축.
  2. 예약 데이터와 고객 특성을 파악하여 **고객 이탈(예약 취소) 여부를 정밀하게 예측** (관측 취소율 37.04%, 유지/취소 약 6:4 비율).
  3. 이탈 위험도가 높은 고객을 조기에 식별하여 **맞춤형 마케팅 및 리텐션 전략 수립 지원** (예: 180일 이상 장기 예약 리워드, 여행사 블록 관리 등).
  4. 현업에서 즉시 활용할 수 있는 **Streamlit 기반 대시보드 및 예측 서비스 구축**.


### 👥 팀원별 역할 설정

| 이름 | 역할 | 담당 업무 |
| :--- | :--- | :--- |
| **오원아** | 팀장 | 데이터 전처리, 모델링 |
| **강유나** | 팀원 | 데이터 전처리, 모델링, 호텔 고객 예약 취소율 조회 페이지 구현 |
| **손혜민** | 팀원 | 데이터 전처리, 모델링 |
| **윤성민** | 팀원 | 데이터 전처리, 모델링 |
| **한승아** | 팀원 | 데이터 전처리, 모델링 |

<br/>

---

## 2. 데이터 전처리 및 EDA

### 📊 데이터 특성 분석
- **데이터셋 출처**: Kaggle - Hotel Booking Demand (Antonio, Almeida & Nunes, 2019)
- **데이터 규모**: 총 **119,390행 × 32열**
- **분석 단위**: 예약 1건 = 1행 (원본에 예약번호가 없어 원본 행 번호를 `source_row_id`로 부여해 추적)
- **Target 정의 (`is_canceled`)**: 
  * `0` (유지/체크아웃): 75,166건 (63.0%)
  * `1` (취소/노쇼): 44,224건 (37.0%)

### 🛠 데이터 정제 및 전처리
#### 1. 공통 전처리 단계
모든 모델에 공통으로 적용하기 위해 원본 파일을 수정하지 않고 정제된 데이터를 생성한 단계입니다.

**[결측치 처리]**
| 대상 컬럼 | 진단 내용 및 상태 | 처리 방법 (Action) | 처리 근거 및 상세 내용 |
| :--- | :--- | :--- | :--- |
| `children` | 4건 (0.003%) | 0으로 채움 | 결측 건수가 매우 적고 어린이 0명인 특성 반영 |
| `country` | 488건 (0.4%) | `"Unknown"`으로 대체 | 국적 미기재 데이터도 하나의 독립된 범주로 유지 |
| `agent` | 16,340건 (13.7%) | `"Unknown"`으로 대체 (여행사 없음) | 여행사 번호는 크기의 의미가 없는 이름표이므로 글자(범주)로 처리 |
| `company` | 112,593건 (94.3%) | `has_company` (있음 1 / 없음 0) 파생 변수 생성 후 원본 삭제 | 값 대부분이 비어 있어 번호 자체보다는 회사 예약 여부 자체가 의미 있음 |

**[이상값 처리]**
| 대상 항목 | 진단 내용 및 상태 | 처리 방법 (Action) | 처리 근거 및 상세 내용 |
| :--- | :--- | :--- | :--- |
| 투숙객 수 | 성인+어린이+유아 합산 0명 (180건) | **제거** (총 182행 제거 중 일부) | 실제로 묵을 사람이 존재하지 않는 잘못된 기록 |
| 1박 요금 (`adr`) | 음수 값 (1건), 5,000 이상 극단값 (1건) | **제거** | 숙박 요금이 음수일 수 없고, €5,400 등 혼자 튀는 값은 입력 오류로 판단 |
| 0박 예약 | 체류 일수가 0인 경우 (715건) | **유지** | 당일 이용(Day-use) 등 실제 예약일 수 있으므로 일괄 삭제 시 패턴 왜곡 방지 |

**[중복 행 처리]**
| 항목 | 진단 내용 및 상태 | 처리 방법 (Action) | 처리 근거 및 상세 내용 |
| :--- | :--- | :--- | :--- |
| 전체 데이터 | 완전히 같은 행 31,994행 (26.8%) | **삭제하지 않고 유지** | 단체 및 여행사에서 같은 조건으로 여러 객실을 한꺼번에 예약한 실제 상황으로 판단 |
| 동일 입력 그룹 | 같은 입력값끼리 묶인 84,668개 그룹 | `group_id` 부여 및 그룹 단위 분할 적용 | 동일 그룹이 Train과 Test에 나뉘어 데이터 누수가 발생하는 것을 방지 (`StratifiedGroupKFold`) |

**[누수 컬럼 제거]**
| 대상 컬럼 | 진단 내용 및 상태 | 처리 방법 (Action) | 처리 근거 및 상세 내용 |
| :--- | :--- | :--- | :--- |
| `reservation_status` | 최종 예약 상태 (Canceled, Check-Out 등) | **제거** | 취소면 1, 체크아웃이면 0으로 정답을 그대로 담고 있는 데이터 자체이므로 삭제 |
| `reservation_status_date` | 상태가 바뀐 최종 날짜 (취소일 등) | **제거** | 결과가 확정된 이후의 시점에 생성되는 정보 |
| `assigned_room_type` | 체크인할 때 실제로 배정된 객실 | **제거** | 예약 확정 시점에는 알 수 없는 체크인 이후의 정보 |
| `booking_changes` | 예약 확정 후 체크인 전까지의 변경 횟수 | **제거** | 예약 이후 발생하는 변화를 담고 있어 예측 시점의 정보에 위배 |
| `required_car_parking_spaces` | 주차 공간 요청 수 (취소율 0%) | **제거** | 데이터 패턴 상 체크인 때 현장에서 기록된 값으로 판단되어 누수 방지를 위해 배제 |

---

#### 2. 모델별 처리 (Pipeline 내부 전처리)
전체 데이터에 일괄 적용하지 않고, 교차 검증 Fold마다 학습 Fold 기준으로만 전처리를 학습(`fit`)하여 데이터 누수를 완벽히 차단했습니다.

* **범주형 인코딩**: 
  * `OneHotEncoder(handle_unknown="ignore")` 적용 (실서비스 시 미지의 범주 유입 대응)
  * 선형 모델의 경우 희소 카테고리 제어를 위해 `min_frequency=100` 설정
* **수치형 스케일링**: 
  * 수치 규모가 모델 가중에 직접 영향을 주는 선형 모델(로지스틱 회귀 등)에 한해 `StandardScaler` 적용

### 💡 핵심 EDA
- **예약 기간 (`lead_time`)**: 예약 후 체크인까지의 기간이 길어질수록 취소율이 계단식으로 증가 (1주 이내 9.6% → 1년 초과 67.7%)
<img width="1080" height="645" alt="image" src="https://github.com/user-attachments/assets/0894b4fc-4101-4eb7-904c-2306672ac83a" />

- **보증금 유형 (`deposit_type`)**: 환불 불가(Non Refund) 조건의 취소율이 99.4%에 달하나, 이는 여행사·단체가 미리 잡아둔 블록 물량이 일괄 취소되는 패턴에 기인함
<img width="1080" height="397" alt="image" src="https://github.com/user-attachments/assets/0eb6ab90-f6a9-4131-bd12-9741b1d846e2" />

- **국적 (`country`)**: 포르투갈 국내 고객의 취소율(56.6%)이 해외 고객(23.6%)의 2배 이상을 기록
<img width="1080" height="315" alt="image" src="https://github.com/user-attachments/assets/3ef4c947-f6c2-4eee-9e62-75741cf1af96" />

- **특별 요청 사항**: 특별 요청이 1개 이상 있는 경우 취소율이 눈에 띄게 감소 (0개 47.7% vs 1개 이상 16~22%)
<img width="1080" height="480" alt="image" src="https://github.com/user-attachments/assets/7de1066c-5668-4227-93c4-3d1ef01dab5d" />

<br/>

---

## 3. 모델링 및 결과 해석

### 사용 모델 및 최종 모델 선정 방식
- **사용 모델 (5종)**: 결정 트리 (Decision Tree), 로지스틱 회귀 (Logistic Regression), 랜덤 포레스트 (Random Forest), XGBoost, Stacking
- **모델 평가 지표**: 모델별 예측 성능과 변별력을 다각도로 검증하기 위해 Accuracy, Precision, Recall, F1-score, ROC-AUC 지표를 활용하여 가장 높은 성능을 가진 모델을 최종 모델로 선정
---

### 2. 모델별 하이퍼파라미터 튜닝 및 최적화
각 모델별로 담당자를 배정하여 `RandomizedSearchCV` 및 `GridSearchCV`를 통해 하이퍼파라미터 튜닝을 거쳐 최적의 설정값을 탐색

### ② 로지스틱 회귀 (Logistic Regression)
* **최적 설정**: `StandardScaler` + `log1p`(치우친 5개 수치형 컬럼) + L1 규제, $C = 0.114$, `class_weight='balanced'`, 임계값 $0.45$   
* **설명**: 다중공선성이 우려되는 파생변수를 제외하고 치우친 수치형 변수에 로그 변환을 적용한 뒤 L1 규제를 통해 성능 개선

### ① 결정 트리 (Decision Tree)
* **최적 설정 및 튜닝**: `max_depth=10`, `min_samples_split=5`, `min_samples_leaf=4`, `criterion='gini'`
* **설명**: 단일 트리의 복잡도를 제어하여 과적합을 방지하고자 최대 깊이와 리프 노드의 최소 샘플 수를 제한하여 최적의 파라미터 도출

### ③ 랜덤 포레스트 (Random Forest)
* **최적 설정**:
  * 베이스 모델 설정: `n_estimators=50`, `max_depth=18`, `min_samples_leaf=2`, `max_features="sqrt"`
  * 고도화 최적 파라미터 (`RandomizedSearchCV`): `max_depth=10`, `min_samples_leaf=1` (교차 검증 최고 ROC-AUC: $0.9064$)
* **설명**: `StratifiedGroupKFold`를 활용하여 학습과 테스트 데이터 간의 그룹 중복을 방지하고, 학습 데이터 내부의 그룹 기반 교차 검증을 통해 과적합을 억제하면서 안정적인 비선형 패턴을 학습하도록 최적의 하이퍼파라미터 탐색

### ④ XGBoost
* **최적 설정**: `n_estimators=357`, `max_depth=7`, `learning_rate=0.1402`, `colsample_bytree=0.6571`, `subsample=0.6571` (평가용 데이터 성능 - Accuracy: $0.8644$, ROC-AUC: $0.9407$)
* **설명**: `StratifiedGroupKFold` 교차 검증과 `logloss` 평가지표를 적용한 `RandomizedSearchCV`를 통해 학습률, 트리 최대 깊이, 샘플링 비율 등을 최적화하였으며, 이를 통해 그래디언트 부스팅 모델의 과적합을 효과적으로 억제하고 예측 성능 극대화

### ⑤ Stacking A
* **최적 설정**: Base Models (XGBoost + Random Forest + Logistic Regression) + Meta-Model 조합   
* **설명**: 트리 기반 앙상블 모델들과 선형 모델(Logistic Regression)의 예측 확률값들을 베이스 피처로 결합하여 메타 모델이 학습하도록 구성

### ⑥ Stacking B
* **최적 설정**: Base Models (XGBoost + Random Forest + MLP) + Meta-Model 조합
* **설명**: 강력한 트리 앙상블 모델들과 다층 인공신경망(MLP)의 예측 확률값들을 베이스 피처로 활용하여, 메타 모델이 최종 예측의 시너지를 극대화하도록 구성

---

### 3. 최종 모델 성능 비교 및 결과 해석
첨부된 성능 비교 데이터를 반영하여 모델별 지표를 정리한 결과입니다.

#### 📊 모델별 성능 비교 요약표

| 모델 | 담당자 | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 강유나 | 0.8086 | 0.7080 | 0.8234 | 0.7614 | 0.9006 |
| **Decision Tree** | 윤성민 | 0.8373 | 0.8123 | 0.7299 | 0.7689 | 0.9159 |
| **Random Forest** | 오원아 | 0.7892 | 0.9609 | 0.4498 | 0.6128 | 0.9095 |
| **XGBoost** | 손혜민 | 0.8644 | 0.8514 | 0.7685 | 0.8078 | 0.9407 |
| **Stacking B** | 한승아 | **0.8658** | **0.8493** | **0.7755** | **0.8108** | **0.9402** |
| **Stacking A** | 한승아 | 0.8650 | 0.8501 | 0.7710 | 0.8085 | 0.9395 |

---

### 🔍 최종 결과 인사이트
1. **정확도(Accuracy) 및 전반적 성능 우위 (Stacking B)**: Stacking B 모델은 정확도($0.8658$), 재현율($0.7755$), F1-score($0.8108$) 부문에서 전반적으로 가장 균형 잡히고 높은 수치를 기록하며 최종적으로 우수한 성능 입증
2. **앙상블 시너지를 통한 안정성 확보**: Stacking A와 Stacking B 모두 단일 모델들의 단점을 보완하고 지표 전반에서 안정적인 고성능을 도출해 냄으로써, 복잡한 호텔 예약 데이터 특성에 앙상블 기법이 효과적이었음을 확인
