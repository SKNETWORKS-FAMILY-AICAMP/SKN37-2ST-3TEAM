# SKN37-2ST-3TEAM

# 🏨 고객 이탈 예측 서비스 (Customer Churn Prediction App)

> **호텔 예약 취소 및 고객 이탈 가능성을 예측하고, 데이터 기반의 리텐션 전략을 제안하는 Streamlit 웹 애플리케이션 서비스입니다.**

<br/>

## 🛠 기술 스택 (Tech Stack)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F79939?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EB2A12?style=for-the-badge&logo=xgboost&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)

<br/>

---

## 📌 목차 (Table of Contents)
1. [프로젝트 설계](#1-프로젝트-설계)
2. [데이터 전처리 및 EDA](#2-데이터-전처리-및-eda)
3. [모델링](#3-모델링)
4. [Streamlit 앱 개발](#4-streamlit-앱-개발)
5. [모델 평가 및 해석](#5-모델-평가-및-해석)

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
1. **결측치 및 관리용 컬럼 처리**:
   * `children` (4건): 0으로 채움
   * `country` (488건), `agent` (16,340건): `"Unknown"` 범주로 처리하여 정보 손실 방지
   * `company` (94.3% 결측): 회사 유무를 나타내는 파생 변수 `has_company`(0/1)로 변환 후 원본 컬럼 삭제
   * 원본 추적을 위한 `source_row_id` 부여
2. **이상값(Outlier) 정제**:
   * 투숙객 0명인 기록(180건), 1박 요금(adr) 음수 및 5,000 이상인 극단적 입력 오류(2건) 등 총 182행 제거 (IQR 일괄 삭제로 인한 패턴 왜곡 방지)
3. **데이터 누수(Data Leakage) 컬럼 완벽 제거**:
   * 예약 확정 시점에 알 수 없는 미래 정보인 `reservation_status`, `reservation_status_date`, `assigned_room_type`, `booking_changes`, `required_car_parking_spaces` 5개 컬럼 제거
4. **중복 행 및 데이터 분할 전략 (`StratifiedGroupKFold`)**:
   * 완전히 동일한 31,994행의 중복 데이터는 실제 단체/여행사 블록 예약으로 간주하여 **삭제하지 않고 유지**
   * 동일 입력값 그룹에 `group_id`를 부여하고, Train/Test 간 데이터 누수를 막기 위해 **`StratifiedGroupKFold`를 적용하여 그룹 단위로 분할** (Train 95,367행 / Test 23,841행)
5. **파생 변수 및 스케일링**:
   * `total_guests` (성인+어린이+유아), `total_nights` (주말+평일 박 수) 생성
   * 선형 모델을 위한 `StandardScaler` 및 왜도가 높은 5개 수치형 컬럼(`lead_time`, `adr` 등) `log1p` 변환 적용

### 💡 핵심 EDA 인사이트 요약
- **예약 기간 (`lead_time`)**: 예약 후 체크인까지의 기간이 길어질수록 취소율이 계단식으로 증가 (1주 이내 9.6% → 1년 초과 67.7%)
- **보증금 유형 (`deposit_type`)**: 환불 불가(Non Refund) 조건의 취소율이 99.4%에 달하나, 이는 여행사·단체가 미리 잡아둔 블록 물량이 일괄 취소되는 패턴에 기인함
- **국적 (`country`)**: 포르투갈 국내 고객의 취소율(56.6%)이 해외 고객(23.6%)의 2배 이상을 기록
- **특별 요청 사항**: 특별 요청이 1개 이상 있는 경우 취소율이 눈에 띄게 감소 (0개 47.7% vs 1개 이상 16~22%)
<br/>

---

## 3. 모델링

### 🤖 모델 후보 선정
고객 이탈 예측을 위한 다양한 알고리즘을 비교 및 검증했습니다.
* **Basic XGBoost**: 기본 파라미터 적용 베이스라인 모델
* **GridSearchCV XGBoost**: 하이퍼파라미터 격자 탐색 적용
* **RandomizedSearchCV XGBoost**: 무작위 하이퍼파라미터 조합 탐색으로 최적화

### ⚙️ 하이퍼파라미터 튜닝
`RandomizedSearchCV`를 통해 탐색한 최적 하이퍼파라미터 세팅:
```python
param_distributions = {
    'n_estimators': 357,
    'max_depth': 7,
    'learning_rate': 0.140,
    'subsample': 0.76,
    'colsample_bytree': 0.65
}
