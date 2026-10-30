# SKN37-2ST-3TEAM

# 🏨 고객 이탈 예측 서비스 (Customer Churn Prediction App)

> **호텔 예약 취소 및 고객 이탈 가능성을 예측하고, 데이터 기반의 리텐션 전략을 제안하는 Streamlit 웹 애플리케이션 서비스입니다.**

<br/>

## 🛠 기술 스택 (Tech Stack)

<!-- 사용한 주요 기술 스택의 배지를 추가하세요. -->
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

<!-- 비즈니스 문제와 프로젝트의 핵심 목표를 명확히 작성합니다. -->
### 🎯 비즈니스 목표 및 문제 정의
* **문제 배경**: 호텔 업계에서 예약 취소(이탈)는 객실 점유율 저하 및 직간접적인 매출 손실을 유발합니다.
* **프로젝트 목표**: 
  1. 예약 데이터와 고객 특성을 파악하여 **고객 이탈(예약 취소) 여부를 정밀하게 예측**.
  2. 이탈 위험도가 높은 고객을 조기에 식별하여 **맞춤형 마케팅 및 리텐션 전략 수립 지원**.
  3. 현업에서 즉시 활용할 수 있는 **Streamlit 기반 대시보드 및 예측 서비스 구축**.

<!-- 팀원별 이름, 역할, 담당한 주요 기여 사항을 정리합니다. -->
### 👥 팀원별 역할 설정

| 이름 | 역할 | 담당 업무 |
| :--- | :--- | :--- |
| **홍길동** | 팀장 / ML Engineeer | 프로젝트 총괄, 데이터 전처리, XGBoost 모델링 및 파이프라인 구축 |
| **김철수** | Data Analyst | 데이터 EDA, 이상치 및 결측치 처리, SHAP 기반 모델 해석 |
| **이영희** | Frontend / App | Streamlit 기반 UI/UX 개발, 모델 연동 및 대시보드 시각화 |

<br/>

---

## 2. 데이터 전처리 및 EDA

<!-- 데이터의 수집 출처와 주요 컬럼의 특성을 기술합니다. -->
### 📊 데이터 특성 분석
* **데이터셋 출처**: Kaggle Hotel Booking Demand Dataset
* **데이터 규모**: 총 119,390건의 예약 데이터 및 32개 피처(Feature)
* **주요 변수**:
  * `is_canceled` (Target): 예약 취소 여부 (0: 유지, 1: 취소)
  * `lead_time`: 예약일과 입실일 사이의 기간
  * `deposit_type`: 보증금 유형 (No Deposit, Non Refund, Editable)
  * `customer_type`: 고객 유형 (Transient, Contract 등)

<!-- 결측치, 이상치, 인코딩, 스케일링 등 전처리 과정을 설명합니다. -->
### 🛠 데이터 전처리
1. **결측값 처리**: 
   * `agent`, `company`: 미지정을 의미하는 'Unknown'으로 대체
   * `children`, `country`: 최빈값(Mode) 및 중앙값(Median)으로 임퓨팅
2. **이상값(Outlier) 처리**: 
   * `ADR` (평균 객실 요금) 비정상 수치(음수 및 극단값) 제거
3. **인코딩 & 스케일링**:
   * 범주형 변수: `OneHotEncoder` (희소 카테고리는 `min_frequency` 설정하여 그룹화)
   * 수치형 변수: `StandardScaler` 적용
4. **학습 및 테스트 데이터 분리**:
   * Data Leakage 방지를 위해 전처리 파이프라인(`ColumnTransformer`) 적용 후 분리
   * `StratifiedGroupKFold`를 활용한 8:2 데이터 분할 (`X_train`: 95,367건, `X_test`: 23,841건)

<br/>

---

## 3. 모델링

<!-- 실험한 머신러닝/딥러닝 모델과 하이퍼파라미터 튜닝 기법을 정리합니다. -->
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
