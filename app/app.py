"""
호텔 예약 취소 위험 관리 앱 (Streamlit)

실행:  streamlit run app.py
필요한 파일 (app.py와 같은 폴더):
  - best_stacking_modelB2.joblib : 스태킹 B 모델 (XGBoost · RF · MLP + 메타 모델)
  - hotel_preprocessed.csv   : 공통 전처리 결과
  - app_utils.py             : 계산 로직

화면 4개 (호텔 직원이 아침에 쓰는 순서)
  1. 오늘 현황   2. 예약 조회   3. 새 예약 예측   4. 취소 위험 대상
입력한 예약은 저장하지 않아요. 내려받기(CSV)만 제공해요.
"""
import os
from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st

import app_utils as u

st.set_page_config(page_title="예약 취소 위험 관리", layout="wide", initial_sidebar_state="expanded")

PAGES = ["홈", "오늘 현황", "예약 조회", "취소 위험 대상", "프로모션 대상", "새 예약 예측"]
DATE_PAGES = ["오늘 현황", "예약 조회", "취소 위험 대상", "프로모션 대상"]   # 기준일을 쓰는 페이지
ICONS = {"홈": ":material/home:", "오늘 현황": ":material/today:", "예약 조회": ":material/search:",
         "새 예약 예측": ":material/calculate:", "취소 위험 대상": ":material/call:",
         "프로모션 대상": ":material/redeem:"}
RISK_STYLE = {"높음": "background-color:#FDE2E1;color:#B0141F;font-weight:700",
              "보통": "background-color:#FDF1D6;color:#8A5A00;font-weight:700",
              "낮음": "background-color:#E2F3E9;color:#1E6B43;font-weight:700"}
EXAMPLE_ID = 36416   # 새 예약 예측 화면의 처음 예시 예약

# ------------------------------------------------------------
# 화면 꾸미기 (메인 #2563EB 파랑 · 배경 #EFF6FF · 흰 카드)
#   st.container(key="card_...") → 흰 카드
#   st.container(key="kpi_hi...") → 위험 높음 강조(빨간 테두리)
# ------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600;700&display=swap');
html, body, .stApp, .stApp p, .stApp label, .stApp button, .stApp input, .stApp li, .stApp h1, .stApp h2, .stApp h3 {
  font-family: 'IBM Plex Sans KR', 'Malgun Gothic', sans-serif !important; }
html { font-size: 14px; }   /* 전체 크기: 16px(기본) → 14px, 숫자를 바꾸면 전체가 같이 커지고 작아져요 */
.stApp { background: #EFF6FF; color: #1E293B; }
[data-testid="stVerticalBlock"] { gap: 0.8rem; }
[data-testid="stSidebarHeader"] { height: 2rem; padding: 0.4rem 1rem 0; }
[data-testid="stSidebarUserContent"] { padding-top: 0.2rem; }
[data-testid="stHeader"] { background: transparent; }
/* 오른쪽 위 메뉴·Deploy 버튼만 숨기고, 사이드바 열기 버튼(≫)은 남겨 둠 */
[data-testid="stDecoration"], [data-testid="stMainMenu"], [data-testid="stAppDeployButton"],
[data-testid="stToolbarActions"] { display: none !important; }
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] {
  display: flex !important; visibility: visible !important; opacity: 1 !important; }
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1400px; }
h1 { font-size: 22px !important; font-weight: 700 !important; padding: 0 !important; line-height: 1.4 !important; margin-bottom: 2px !important; }
h3 { font-size: 15px !important; font-weight: 700 !important; padding: 0 0 6px !important; line-height: 1.4 !important; }
.page-sub { color: #64748B; margin: 0 0 10px; font-size: 13px; line-height: 1.5; }

/* ---------- 사이드바 (흰 바탕 · 파랑 강조) ---------- */
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #DBEAFE; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #1E293B; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] input:disabled,
[data-testid="stSidebar"] [data-baseweb="input"] input {
  color: #1E293B !important; -webkit-text-fill-color: #1E293B !important;
  opacity: 1 !important; font-weight: 700 !important; font-size: 13px !important; }
[data-testid="stSidebar"] [data-baseweb="input"] { background: #FFFFFF !important; }
[data-testid="stSidebar"] [data-baseweb="input"] *, [data-testid="stSidebar"] [data-baseweb="input"] *::placeholder {
  color: #1E293B !important; -webkit-text-fill-color: #1E293B !important; opacity: 1 !important; }
[data-testid="stSidebar"] [data-testid="stSliderThumbValue"] { color: #1D4ED8 !important; font-weight: 700; }
.sb-brand { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 16px;
  line-height: 1.3; color: #1E293B; padding: 0 4px 16px; margin-bottom: 14px; border-bottom: 1px solid #DBEAFE; }
/* 기준일 날짜 칸: 흰 바탕 + 진한 글자 (칸 안의 모든 요소에 강제 적용) */
[data-testid="stSidebar"] [data-testid="stDateInput"] *:not(label):not(label *) {
  color: #1E293B !important; -webkit-text-fill-color: #1E293B !important; opacity: 1 !important; }
[data-testid="stSidebar"] [data-testid="stDateInput"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-testid="stDateInput"] [data-baseweb="input"] * {
  background-color: #F8FAFF !important; }
[data-testid="stSidebar"] [data-testid="stDateInput"] input { font-weight: 700 !important; }
[data-testid="stSidebar"] button { justify-content: flex-start !important; border: none !important;
  border-radius: 8px !important; min-height: 36px; padding-top: 0 !important; padding-bottom: 0 !important; background: transparent !important; box-shadow: none !important; }
[data-testid="stSidebar"] button > div { justify-content: flex-start !important; width: 100%; }
[data-testid="stSidebar"] button p { color: #334155 !important; font-size: 14px; text-align: left; padding-left: 4px; }
[data-testid="stSidebar"] button:hover { background: #E6EFFE !important; }
[data-testid="stSidebar"] button [data-testid="stIconMaterial"] { color: #1D4ED8 !important; }
[data-testid="stSidebar"] button[kind="primary"] [data-testid="stIconMaterial"],
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] [data-testid="stIconMaterial"] { color: #FFFFFF !important; }
[data-testid="stSidebar"] button[kind="primary"],
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] { background: #2563EB !important; }
[data-testid="stSidebar"] button[kind="primary"] p,
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] p { color: #FFFFFF !important; font-weight: 700; }

/* ---------- 기준일 바 ---------- */
.st-key-card_datebar { padding: 10px 16px 12px !important; }
.datebar-note { color: #64748B; font-size: 12.5px; margin: 0 0 8px; }
[data-testid="stMain"] [data-testid="stDateInput"] input { font-weight: 700 !important; color: #1E293B !important; -webkit-text-fill-color: #1E293B !important; }

/* ---------- 카드 ---------- */
[class*="st-key-card"] { background: #FFFFFF; border: 1px solid #DBE4F0; border-radius: 10px; padding: 16px 18px; }
[class*="st-key-card"] [data-testid="stCaptionContainer"] { margin-top: -6px; }
[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #DBE4F0; border-radius: 10px; padding: 10px 14px; }
[data-testid="stMetric"] > div { gap: 4px; }
[data-testid="stMetricLabel"] p { font-size: 12px !important; color: #64748B !important; }
[data-testid="stMetricValue"] { font-size: 21px !important; font-weight: 700; color: #1E293B; }
[data-testid="stMetricDelta"] { color: #64748B !important; background: transparent !important; padding: 0 !important; }
[data-testid="stMetricDelta"] svg { display: none; }
[data-testid="stMetricDelta"] div { font-size: 11px; }
[class*="st-key-card"] [data-testid="stMetric"] { border: none; padding: 0; background: transparent; }

/* 위험 높음 강조 */
[class*="st-key-kpi_hi"] [data-testid="stMetric"] { border: 2px solid #C1121F; padding: 9px 13px; }
[class*="st-key-kpi_hi"] [data-testid="stMetricLabel"] p,
[class*="st-key-kpi_hi"] [data-testid="stMetricValue"] { color: #B0141F !important; font-weight: 700; }
.st-key-card_result_hi { border: 2px solid #C1121F; }
.st-key-card_result_hi [data-testid="stMetricValue"] { color: #B0141F; font-size: 34px !important; }
.st-key-card_result [data-testid="stMetricValue"] { font-size: 34px !important; }

/* 오늘 할 일 (파랑 강조 카드) */
.st-key-dark_todo { background: #E6EFFE; border: 1px solid #BFD3FB; border-radius: 10px; padding: 16px 18px; }
.st-key-dark_todo li { margin-bottom: 4px; }
.st-key-dark_todo h3 { color: #1E40AF !important; }
.st-key-dark_todo p, .st-key-dark_todo li { color: #1E293B !important; }
.st-key-dark_todo strong { color: #1E40AF; }
.st-key-dark_todo button[kind="secondary"], .st-key-dark_todo button[data-testid="stBaseButton-secondary"] {
  background: #FFFFFF !important; border: 1px solid #BFD3FB !important; }

/* 오늘 현황 (다른 페이지보다 크게) */
.st-key-today h1 { font-size: 26px !important; }
.st-key-today .page-sub { font-size: 15px; }
.st-key-today h3 { font-size: 18px !important; }
.st-key-today [data-testid="stMetric"] { padding: 16px 20px; }
.st-key-today [class*="st-key-kpi_hi"] [data-testid="stMetric"] { padding: 15px 19px; }
.st-key-today [data-testid="stMetricLabel"] p { font-size: 14px !important; }
.st-key-today [data-testid="stMetricValue"] { font-size: 30px !important; }
.st-key-today [data-testid="stMetricDelta"] div { font-size: 13px; }
.st-key-today [data-testid="stCaptionContainer"] p { font-size: 14px; }
.st-key-today [class*="st-key-card"] { padding: 22px 24px; }
.st-key-today .st-key-dark_todo { padding: 22px 24px; }
.st-key-today .st-key-dark_todo li { font-size: 16px; margin-bottom: 8px; }
.st-key-today .st-key-dark_todo button { min-height: 46px; }
.st-key-today [data-testid="stSegmentedControl"] button p { font-size: 14px; }
.st-key-today [data-testid="stTable"] table { width: 100%; }
.st-key-today [data-testid="stTable"] th, .st-key-today [data-testid="stTable"] td { font-size: 15px; padding: 10px 14px; }

/* 홈 (홈 화면은 여유가 있어서 다른 페이지보다 크게) */
.hero { margin-bottom: 18px; }   /* 배너와 카드 줄 사이 간격 */
[class*="st-key-card_home"] { margin-bottom: 14px; }   /* 카드 줄과 아래 안내 박스 사이 간격 */
.hero { background: linear-gradient(120deg, #1E3A8A 0%, #1D4ED8 50%, #2563EB 100%); border-radius: 14px;
  padding: 25px 38px; color: #FFFFFF; }
.hero h2 { color: #FFFFFF !important; font-size: 30px !important; font-weight: 700; margin: 0 0 8px; padding: 0 !important; }
.hero p { color: #E6EFFE !important; margin: 0; font-size: 16px; }
.hero .tags { margin-top: 18px; display: flex; flex-wrap: wrap; gap: 8px; }
.hero .tags span { background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.45);
  border-radius: 999px; padding: 4px 14px; font-size: 13px; color: #FFFFFF; }
.home-h { font-size: 18px; font-weight: 700; margin: 12px 0 0; }
[class*="st-key-card_home"] { padding: 22px 24px !important; }
.feat-title { font-weight: 700; font-size: 17px; margin: 0 0 6px; }
.feat-desc { color: #64748B; font-size: 14px; line-height: 1.7; min-height: 6.8em; }   /* 4줄 높이로 고정 → 바로 가기 버튼 위치가 같아짐 */
[class*="st-key-card_home"] button { min-height: 42px; }
.info { background: #F5F9FF; border: 1px solid #BFD3FB; border-radius: 12px; padding: 22px 26px;
  font-size: 14.5px; line-height: 2; height: 100%; }
.info b { color: #1E40AF; }
.info .t { font-size: 16px; }
.step { display: flex; gap: 12px; align-items: center; font-size: 14px; line-height: 1.6; margin-bottom: 12px; }
.step .n { flex: 0 0 26px; height: 26px; border-radius: 50%; background: #2563EB; color: #FFFFFF;
  font-size: 13px; font-weight: 700; display: flex; align-items: center; justify-content: center; }
.steps-t { font-size: 16px; font-weight: 700; margin-bottom: 12px; }

/* 위험 등급 배지 */
.chip { display: inline-block; padding: 1px 10px; border-radius: 999px; font-weight: 700; font-size: 12px; }
.chip-높음 { background: #FDE2E1; color: #B0141F; }
.chip-보통 { background: #FDF1D6; color: #8A5A00; }
.chip-낮음 { background: #E2F3E9; color: #1E6B43; }
.up { color: #B0141F; font-weight: 700; } .dn { color: #1E6B43; font-weight: 700; }
.msg { background: #EFF6FF; border: 1px solid #DBE4F0; border-radius: 10px; padding: 14px 16px;
  white-space: pre-line; font-size: 14px; line-height: 1.7; }
</style>
"""
ICON = ('<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="1.8">'
        '<rect x="4" y="3" width="16" height="18" rx="1.5"></rect>'
        '<path d="M8 7h2M14 7h2M8 11h2M14 11h2M8 15h2M14 15h2M11 21v-3h2v3"></path></svg>')
st.markdown(CSS, unsafe_allow_html=True)


def header(title, sub):
    st.title(title)
    st.markdown(f'<p class="page-sub">{sub}</p>', unsafe_allow_html=True)




# ------------------------------------------------------------
# 데이터 · 모델 불러오기 (한 번만 계산하고 저장해 둠)
# ------------------------------------------------------------
@st.cache_resource
def get_model(mtime):
    return u.load_model(u.MODEL_PATH)


@st.cache_data
def get_data():
    return u.load_data(u.DATA_PATH)


@st.cache_data
def get_predictions(mtime):
    df = get_data()
    return u.add_predictions(get_model(mtime), df[df["split"] == "test"])


@st.cache_data
def get_background(mtime):
    df = get_data()
    feats = u.model_features(get_model(mtime))
    train = df[df["split"] == "train"]
    return u.background_sample(train[feats]), u.baseline_values(train[feats])


@st.cache_data(show_spinner="주요 요인 계산 중…")
def get_window(base_day, days, mtime):
    """기준일부터 N일 동안 체크인하는 예약 + 예약별 주요 요인"""
    model = get_model(mtime)
    w = u.upcoming(get_predictions(mtime), base_day, days)
    if w.empty:
        w["주요 요인"] = []
        return w
    bg, _ = get_background(mtime)
    eff = u.explain(model, w[u.model_features(model)], bg)
    w["주요 요인"] = [u.reason_text(w.loc[i], eff.loc[i]) for i in w.index]
    return w


if not os.path.exists(u.MODEL_PATH):
    st.error(f"모델 파일 `{u.MODEL_PATH}`이 없어요. 모델 파일은 용량이 커서 깃허브에 올리지 않았어요. "
             f"README의 구글 드라이브 링크에서 받아 프로젝트 폴더(app 폴더 바깥)에 넣은 뒤, "
             f"그 폴더에서 `python -m streamlit run app/app.py`로 실행해 주세요.")
    st.stop()

MTIME = os.path.getmtime(u.MODEL_PATH)   # 모델 파일을 바꾸면 계산을 새로 함
model = get_model(MTIME)
FEATS = u.model_features(model)
USES_YEAR = "arrival_date_year" in FEATS
data = get_data()
preds = get_predictions(MTIME)
test_start = preds["arrival_date"].min().date()
test_end = preds["arrival_date"].max().date()


# ------------------------------------------------------------
# 화면 이동 (버튼에서 다른 페이지로)
# ------------------------------------------------------------
if st.session_state.get("page") not in PAGES:   # 처음 열었거나, 이름이 바뀐 페이지에 있었으면 홈으로
    st.session_state.page = PAGES[0]
if "form_ver" not in st.session_state:
    st.session_state.form_ver = 0


def go(page, prefill=None):
    st.session_state.page = page
    if prefill is not None:
        st.session_state.prefill = prefill
        st.session_state.form_ver += 1   # 입력 칸을 새 값으로 다시 그리기


def styled(df, risk_col="위험 등급"):
    return df.style.map(lambda v: RISK_STYLE.get(v, ""), subset=[risk_col])


# ------------------------------------------------------------
# 사이드바
# ------------------------------------------------------------
with st.sidebar:
    st.markdown(f'<div class="sb-brand">{ICON}<div>호텔 예약 관리</div></div>', unsafe_allow_html=True)
    for p in PAGES:   # 메뉴: 지금 페이지는 파란 버튼
        st.button(p, key=f"nav_{p}", icon=ICONS[p], use_container_width=True, on_click=go, args=(p,),
                  type="primary" if st.session_state.page == p else "secondary")


# ------------------------------------------------------------
# 기준일 · 기간 (각 페이지 위쪽 date_bar()에서 고름)
#   위젯 값(base_day_w)은 그 위젯이 없는 페이지에 가면 지워져서, 별도 값(base_day_v)에도 저장해 둠
# ------------------------------------------------------------
if "base_day_v" not in st.session_state:
    st.session_state.base_day_v = max(test_start, min(date(2017, 5, 1), test_end))
    st.session_state.days_v = 14
base_day = st.session_state.get("base_day_w", st.session_state.base_day_v)
days = st.session_state.get("days_w") or st.session_state.days_v   # 칩을 다시 눌러 선택이 풀리면 이전 값 유지
if st.session_state.get("days_w") is None and "days_w" in st.session_state:
    st.session_state.days_w = days
st.session_state.base_day_v, st.session_state.days_v = base_day, days

window = get_window(base_day, days, MTIME)
period = f"{base_day:%m-%d} ~ {(base_day + timedelta(days=days - 1)):%m-%d}"


def date_bar():
    """기준일 · 기간 고르기 (오늘 현황 · 예약 조회 · 취소 위험 대상 · 프로모션 대상 페이지 위쪽)"""
    st.session_state.setdefault("base_day_w", st.session_state.base_day_v)
    st.session_state.setdefault("days_w", st.session_state.days_v)
    with st.container(key="card_datebar"):
        c1, c2, c3 = st.columns([1.3, 1.6, 3], vertical_alignment="bottom")
        c1.date_input("기준일", key="base_day_w", min_value=test_start, max_value=test_end)
        c2.segmented_control("앞으로 체크인 예약", [7, 14, 30], key="days_w",
                             format_func=lambda d: f"{d}일")
        c3.markdown(f'<p class="datebar-note">{period} 체크인 예약 · 기준일은 '
                    f'{test_start:%Y.%m.%d} ~ {test_end:%Y.%m.%d} 사이에서 골라요</p>', unsafe_allow_html=True)


# ============================================================
# 0. 홈
# ============================================================
FEATURES = [
    ("오늘 현황", ":material/today:", "기준일부터 며칠 동안 체크인할 예약의 취소 위험을 한눈에 봐요. 위험 높음 건수·예상 취소·영향 매출."),
    ("예약 조회", ":material/search:", "체크인 예정 예약을 취소 확률 순으로 보고, 예약마다 확률을 높이거나 낮춘 주요 요인을 확인해요."),
    ("취소 위험 대상", ":material/call:", "위험 높음 예약을 단체·여행사와 개인 고객으로 나눠 연락 기한과 예약 확인 문구를 정리해요."),
    ("프로모션 대상", ":material/redeem:", "체크인 확정 리워드(웰컴 드링크·포인트)와 보증금 안내 문자 대상을 고르고 발송해요."),
    ("새 예약 예측", ":material/calculate:", "예약 정보를 입력하면 바로 취소 확률·위험 등급·주요 요인·권장 대응을 보여줘요."),
]


def page_home():
    st.markdown(
        f'''<div class="hero"><h2>호텔 예약 취소 위험 관리</h2>
        <p>예약이 확정된 직후, 취소될 위험을 미리 예측하고 먼저 확인 연락을 할 수 있게 도와줘요.</p>
        <div class="tags"><span>호텔 2곳 (City · Resort)</span><span>2015.07 ~ 2017.08 예약</span>
        <span>예약 데이터 {len(data):,}건</span><span>예측 모델: {u.MODEL_LABEL}</span></div></div>''',
        unsafe_allow_html=True)

    # st.markdown('<div class="home-h">이런 일을 할 수 있어요</div>', unsafe_allow_html=True)
    cols = st.columns(len(FEATURES), gap="medium")
    for col, (page, icon, desc) in zip(cols, FEATURES):
        with col.container(key=f"card_home_{PAGES.index(page)}"):
            st.markdown(f'<div class="feat-title">{page}</div><div class="feat-desc">{desc}</div>',
                        unsafe_allow_html=True)
            st.button("바로 가기", key=f"home_go_{page}", icon=icon, on_click=go, args=(page,),
                      use_container_width=True)

    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            f'''<div class="info"><b class="t">이렇게 판단해요</b><br>
            · <b>예측 시점</b>: 예약이 확정된 직후. 예약 변경·주차 요청처럼 그 뒤에 생기는 정보는 쓰지 않아요.<br>
            · <b>위험 등급</b>: 취소 확률 {u.HIGH:.0%} 이상 <b>높음</b> · {u.THRESHOLD:.0%}~{u.HIGH:.0%} <b>보통</b> · {u.THRESHOLD:.0%} 미만 <b>낮음</b><br>
            · <b>주요 요인</b>: 값 하나를 다른 예약 값으로 바꿔 다시 예측했을 때 확률이 얼마나 달라지는지로 정해요.<br>
            · <b>대응 방법</b>: 예약 유지 확인 연락 + <b>체크인 확정 리워드</b>(웰컴 드링크·포인트) · 보증금 안내 문자<br>
            · <b>금액</b>: 데이터는 유로(€) 기준이라 1€ = {u.EUR_KRW:,}원으로 바꿔 보여줘요.</div>''',
            unsafe_allow_html=True)
    with right.container(key="card_home_steps"):
        st.markdown('<div class="steps-t">사용 순서</div>', unsafe_allow_html=True)
        steps = ["각 페이지 위쪽에서 <b>기준일</b>과 볼 기간을 골라요.",
                 "<b>오늘 현황</b>에서 위험 높음 예약이 몰린 날을 확인해요.",
                 "<b>취소 위험 대상</b>에서 단체여행사와 개인에게 연락해요.",
                 "새로 들어온 예약은 <b>새 예약 예측</b>에서 바로 확인해요."]
        st.markdown("".join(f'<div class="step"><span class="n">{i}</span><span>{t}</span></div>'
                            for i, t in enumerate(steps, 1)), unsafe_allow_html=True)



# ============================================================
# 1. 오늘 현황
# ============================================================
def page_today():
    with st.container(key="today"):
        head, sel = st.columns([3, 2], vertical_alignment="bottom")
        with head:
            header("오늘 현황", f"앞으로 {days}일({period}) 체크인 예정 예약의 취소 위험이에요")
        hotel = sel.segmented_control("호텔", ["두 호텔 전체", "City Hotel", "Resort Hotel"],
                                      default="두 호텔 전체", label_visibility="collapsed") or "두 호텔 전체"
        date_bar()
        w = window if hotel == "두 호텔 전체" else window[window["hotel"] == hotel]
        if w.empty:
            st.info("이 기간에 체크인하는 예약이 없어요. 기준일을 바꿔 보세요.")
            return
        s = u.summary(w)

        c = st.columns(4)
        c[0].metric("체크인 예정 예약", f"{s['total']:,}건",
                    " · ".join(f"{h.replace(' Hotel', '')} {n}" for h, n in w["hotel"].value_counts().items()),
                    delta_color="off")
        with c[1].container(key="kpi_hi_today"):
            st.metric("취소 위험 높음", f"{s['high']:,}건",
                      f"전체의 {s['high'] / s['total']:.0%} · 연락 필요", delta_color="off")
        c[2].metric("예상 취소", f"약 {s['expected_cancel']:,.0f}건",
                    "예약별 취소 확률을 모두 더한 값", delta_color="off")
        c[3].metric("취소 시 영향 매출", f"약 {u.won_short(s['revenue_at_risk'])}",
                    "취소 확률 × 1박 요금 × 박수", delta_color="off")

        with st.container(key="card_daily"):
            st.subheader("날짜별 체크인 예약")
            st.caption("빨간 부분이 취소 위험 높음 예약 수예요")
            daily = (w.assign(구분=w["risk"].map(lambda r: "위험 높음" if r == "높음" else "그 외"))
                      .groupby([w["arrival_date"].dt.strftime("%m-%d"), "구분"]).size()
                      .rename("건수").reset_index().rename(columns={"arrival_date": "체크인 날짜"}))
            chart = (alt.Chart(daily).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                     .encode(x=alt.X("체크인 날짜:N", title=None, axis=alt.Axis(labelAngle=0)),
                             y=alt.Y("건수:Q", title=None),
                             color=alt.Color("구분:N", title=None, legend=alt.Legend(orient="top", labelFontSize=13, symbolSize=120),
                                             scale=alt.Scale(domain=["위험 높음", "그 외"],
                                                             range=["#C1121F", "#BFD3FB"])),
                             order=alt.Order("구분:N", sort="descending"),
                             tooltip=["체크인 날짜", "구분", "건수"])
                     .properties(height=300)
                     .configure_axis(labelFontSize=13, grid=True, gridColor="#E8EEF7", domainColor="#CBD5E1")
                     .configure_view(strokeWidth=0))
            st.altair_chart(chart, use_container_width=True)
            hi_day = daily[daily["구분"] == "위험 높음"].nlargest(2, "건수")["체크인 날짜"].tolist()
            if hi_day:
                st.caption(f"{'과 '.join(hi_day)}에 위험 예약이 몰려 있어요 → 이날 체크인 예약부터 확인하세요.")

        left, right = st.columns(2)
        with left.container(key="card_grade"):
            st.subheader("위험 등급별 예약")
            table = (w.pivot_table(index="hotel", columns="risk", values="prob", aggfunc="size", fill_value=0)
                      .reindex(columns=["높음", "보통", "낮음"], fill_value=0))
            table.index.name = "호텔"
            # 높음 칸만 위험 등급 표와 같은 색으로 강조 (연분홍 바탕 + 빨간 굵은 글씨)
            st.table(table.style.set_properties(subset=["높음"], **{
                "background-color": "#FDE2E1", "color": "#B0141F", "font-weight": "700"}))
            st.caption(f"높음: 취소 확률 {u.HIGH:.0%} 이상 · 보통 {u.THRESHOLD:.0%}~{u.HIGH:.0%} · 낮음 {u.THRESHOLD:.0%} 미만")
        with right.container(key="dark_todo"):
            st.subheader("오늘 할 일")
            blocks, people = u.split_contacts(w, base_day)
            st.markdown(f"- 단체·여행사 블록 **{len(blocks)}곳**에 인원 확정 확인 (예약 {int(blocks['객실'].sum())}건)\n"
                        f"- 개인 고객 위험 높음 **{len(people)}건**에 숙박 확인 안내\n"
                        f"- 보통 **{s['mid']}건**은 체크인 전 리마인드 안내")
            b1, b2 = st.columns(2)
            b1.button("취소 위험 대상 보기", type="primary", on_click=go, args=("취소 위험 대상",), use_container_width=True)
            b2.button("예약 조회", on_click=go, args=("예약 조회",), use_container_width=True)


# ============================================================
# 2. 예약 조회
# ============================================================
def page_bookings():
    header("예약 조회", f"{period} 체크인 예정 예약을 취소 확률이 높은 순서로 보여줘요")
    date_bar()

    with st.container(key="card_filter"):
        f1, f2, f3, f4 = st.columns([2, 1, 1, 1], vertical_alignment="bottom")
        cnt = window["risk"].value_counts()
        risks = f1.pills("위험 등급", ["높음", "보통", "낮음"], selection_mode="multi",
                         default=["높음", "보통", "낮음"],
                         format_func=lambda r: f"{r} {cnt.get(r, 0):,}") or ["높음", "보통", "낮음"]
        hotel = f2.selectbox("호텔", ["전체", "City Hotel", "Resort Hotel"])
        segs = ["전체"] + sorted(window["market_segment"].unique().tolist())
        seg = f3.selectbox("예약 경로", segs)
        q = f4.text_input("예약번호 검색", placeholder="예: 36416")

    w = window[window["risk"].isin(risks)]
    if hotel != "전체":
        w = w[w["hotel"] == hotel]
    if seg != "전체":
        w = w[w["market_segment"] == seg]
    if q.strip():
        w = w[w["source_row_id"].astype(str).str.contains(q.strip())]
    w = w.sort_values("prob", ascending=False)

    show = pd.DataFrame({
        "예약번호": w["source_row_id"].astype(str),
        "체크인 날짜": w["arrival_date"].dt.strftime("%Y-%m-%d"),
        "호텔": w["hotel"].str.replace(" Hotel", ""),
        "예약~체크인(일)": w["lead_time"],
        "보증금": w["deposit_type"],
        "여행사": w["agent"].map(lambda a: "없음" if a == "Unknown" else f"{a}번"),
        "취소 확률": (w["prob"] * 100).round(1),
        "위험 등급": w["risk"],
        "주요 요인": w["주요 요인"],
    })
    with st.container(key="card_table"):
        st.caption(f"{len(show):,}건 · 행을 누르면 아래에 상세가 나와요 · ↑ 취소 가능성을 높인 요인 · ↓ 낮춘 요인")
        event = st.dataframe(styled(show), hide_index=True, use_container_width=True, height=400,
                             on_select="rerun", selection_mode="single-row",
                             column_config={"취소 확률": st.column_config.NumberColumn(format="%.1f%%"),
                                            "주요 요인": st.column_config.TextColumn(width="large")})

    st.download_button("조회 결과 CSV 내려받기", show.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"bookings_{base_day}.csv", mime="text/csv")
    st.caption("내려받기만 하고 앱에는 저장하지 않아요.")

    rows = event.selection.rows if event and event.selection else []
    if rows:
        r = w.iloc[rows[0]]
        with st.container(key="card_detail"):
            st.subheader(f"예약 {r['source_row_id']} · {r['hotel']}")
            a, b = st.columns([1, 2])
            a.metric("취소 확률", f"{r['prob']:.1%}", r["risk"], delta_color="off")
            b.markdown("**권장 대응**\n" + "\n".join(f"1. {t}" for t in u.action_for(r["risk"], r["agent"], r["deposit_type"])))
            st.button("이 예약 값으로 새 예약 예측 열기", on_click=go,
                      args=("새 예약 예측", u.form_from_row(r)))


# ============================================================
# 3. 새 예약 예측
# ============================================================
def options(col):
    return data[col].value_counts().index.tolist()


def default_form():
    ex = data[data["source_row_id"] == EXAMPLE_ID]
    row = ex.iloc[0] if len(ex) else data[data["split"] == "test"].iloc[0]
    return u.form_from_row(row)


def page_predict():
    header("새 예약 예측", "예약 정보를 입력하면 바로 취소 확률을 계산해요 · 입력한 예약은 저장되지 않아요")
    v = st.session_state.get("prefill") or default_form()
    k = lambda name: f"{name}_{st.session_state.form_ver}"   # 입력 칸 key
    _, base = get_background(MTIME)

    if USES_YEAR:
        d_min, d_max = u.DATA_START, u.DATA_END
        st.info("이 모델은 체크인 연도도 입력으로 써서, 학습 데이터 기간(2015.07 ~ 2017.08) 체크인 예약만 받아요.")
    else:
        d_min, d_max = date(2015, 1, 1), date(2030, 12, 31)

    def pick(label, col, key, fmt=None):
        opts = options(col)
        cur = v[key]
        if cur not in opts:
            opts = [cur] + opts
        return st.selectbox(label, opts, index=opts.index(cur), key=k(key),
                            format_func=fmt or (lambda x: x))

    left, right = st.columns([3, 2], gap="large")
    with left.form("booking", border=False), st.container(key="card_form"):
        st.subheader("예약 정보")
        st.markdown("**:blue[숙박]**")
        c = st.columns(3)
        with c[0]:
            hotel = pick("호텔", "hotel", "hotel")
        arrival = c[1].date_input("체크인 날짜", min(max(v["arrival_date"], d_min), d_max),
                                  min_value=d_min, max_value=d_max, key=k("arr"))
        # 예약한 날은 체크인 날짜 기준 최대 2년 전까지만 (학습 데이터 범위)
        arr_now = st.session_state.get(k("arr"), v["arrival_date"])
        book_min = arr_now - timedelta(days=u.MAX_LEAD_DAYS)
        booked = c[2].date_input("예약한 날 (체크인 2년 전까지)", min(max(v["booking_date"], book_min), arr_now),
                                 min_value=d_min - timedelta(days=u.MAX_LEAD_DAYS), max_value=d_max, key=k("book"))
        c = st.columns(3)
        we = c[0].number_input("주말 박", 0, 30, v["stays_in_weekend_nights"], key=k("we"))
        wk = c[1].number_input("평일 박", 0, 60, v["stays_in_week_nights"], key=k("wk"))
        adr_krw = c[2].number_input("1박 요금 (원)", 0, 2_000_000, int(round(v["adr"] * u.EUR_KRW, -3)),
                                    step=10_000, key=k("adr"))
        adr = adr_krw / u.EUR_KRW   # 모델은 유로 기준으로 학습해서 다시 유로로 바꿔 넣음
        c = st.columns(3)
        adults = c[0].number_input("성인", 0, 10, v["adults"], key=k("ad"))
        children = c[1].number_input("어린이", 0, 10, v["children"], key=k("ch"))
        babies = c[2].number_input("유아", 0, 10, v["babies"], key=k("ba"))
        c = st.columns(3)
        with c[0]:
            room = pick("객실 타입", "reserved_room_type", "reserved_room_type")
        with c[1]:
            meal = pick("식사", "meal", "meal")
        req = c[2].number_input("특별 요청 수", 0, 5, v["total_of_special_requests"], key=k("req"))

        st.markdown("**:blue[예약 경로 · 고객]**")
        c = st.columns(3)
        with c[0]:
            deposit = pick("보증금", "deposit_type", "deposit_type")
        with c[1]:
            seg = pick("예약 경로", "market_segment", "market_segment")
        with c[2]:
            chan = pick("유통 채널", "distribution_channel", "distribution_channel")
        c = st.columns(3)
        with c[0]:
            agent = pick("여행사", "agent", "agent", lambda a: "없음(직접 예약)" if a == "Unknown" else f"{a}번")
        with c[1]:
            ctype = pick("고객 유형", "customer_type", "customer_type")
        with c[2]:
            country = pick("국적", "country", "country")
        c = st.columns(3)
        repeated = c[0].selectbox("재방문 고객", [0, 1], index=v["is_repeated_guest"],
                                  format_func=lambda x: "예" if x else "아니오", key=k("rep"))
        prev_c = c[1].number_input("이전 취소 횟수", 0, 30, v["previous_cancellations"], key=k("pc"))
        prev_b = c[2].number_input("이전 이용 횟수", 0, 80, v["previous_bookings_not_canceled"], key=k("pb"))
        c = st.columns(3)
        wait = c[0].number_input("대기 일수", 0, 400, v["days_in_waiting_list"], key=k("wait"))
        company = c[1].selectbox("회사 예약", [0, 1], index=v["has_company"],
                                 format_func=lambda x: "예" if x else "아니오", key=k("comp"))
        st.caption("예약 변경·주차 요청·객실 변경처럼 예약 확정 뒤에 생기는 값은 받지 않아요.")
        submitted = st.form_submit_button("취소 확률 예측", type="primary", use_container_width=True)

    st.button("예시 예약으로 초기화", on_click=go, args=("새 예약 예측", default_form()))

    form = dict(hotel=hotel, arrival_date=arrival, booking_date=booked,
                stays_in_weekend_nights=we, stays_in_week_nights=wk, adr=adr,
                adults=adults, children=children, babies=babies,
                reserved_room_type=room, meal=meal, total_of_special_requests=req,
                deposit_type=deposit, market_segment=seg, distribution_channel=chan,
                agent=agent, customer_type=ctype, country=country, is_repeated_guest=repeated,
                previous_cancellations=prev_c, previous_bookings_not_canceled=prev_b,
                days_in_waiting_list=wait, has_company=company)

    with right:
        if booked > arrival:
            st.error("예약한 날이 체크인 날짜보다 늦어요. 날짜를 확인해 주세요.")
            return
        if (arrival - booked).days > u.MAX_LEAD_DAYS:
            st.error(f"예약~체크인 기간은 최대 2년({u.MAX_LEAD_DAYS}일)까지만 예측할 수 있어요. "
                     f"지금 {(arrival - booked).days}일이에요. 예약한 날을 체크인 날짜와 더 가깝게 바꿔 주세요.")
            return
        if we + wk == 0:
            st.error("숙박 일수가 0박이에요. 주말 박이나 평일 박을 입력해 주세요.")
            return
        if not submitted and "prefill" not in st.session_state and st.session_state.form_ver == 0:
            st.caption("왼쪽에 값을 넣고 **취소 확률 예측**을 눌러 주세요. (예시 예약이 미리 들어가 있어요)")
        x = u.build_input(form, FEATS, base)
        p = float(u.predict(model, x)[0])
        risk = u.risk_label(p)
        with st.container(key="card_result_hi" if risk == "높음" else "card_result"):
            t1, t2 = st.columns([1, 1], vertical_alignment="center")
            t1.subheader("예측 결과")
            t2.markdown(f'<div style="text-align:right"><span class="chip chip-{risk}">취소 위험 {risk}</span></div>',
                        unsafe_allow_html=True)
            st.metric("취소 확률", f"{p:.1%}")
            st.progress(min(p, 1.0), text=f"보통 {u.THRESHOLD:.0%} · 높음 {u.HIGH:.0%}부터")
            bg, _ = get_background(MTIME)
            eff = u.explain(model, x, bg).iloc[0]
            reasons = u.reason_list(x.iloc[0], eff, top=3)
            st.markdown("**주요 요인**")
            items = "".join(f'<li><span class="{"up" if d == "↑" else "dn"}">{d}</span> {t}</li>' for d, t, _ in reasons)
            st.markdown(f'<ul style="list-style:none;padding:0;margin:0;line-height:1.8;font-size:13px">{items or "<li>뚜렷한 요인 없음</li>"}</ul>',
                        unsafe_allow_html=True)

        with st.container(key="card_whatif"):
            st.subheader("이렇게 바뀌면?")
            st.caption("같은 예약에서 한두 값만 바꿔 다시 예측한 결과")
            alt_rows = [("현재", form)]
            alt_rows.append((f"특별 요청 {req} → {req + 2}개", {**form, "total_of_special_requests": req + 2}))
            if (arrival - booked).days > 30:
                short = {**form, "booking_date": arrival - timedelta(days=30)}
                alt_rows.append((f"예약~체크인 {(arrival - booked).days} → 30일", short))
                alt_rows.append(("둘 다 바뀌면", {**short, "total_of_special_requests": req + 2}))
            res = []
            for name, f in alt_rows:
                pp = float(u.predict(model, u.build_input(f, FEATS, base))[0])
                res.append({"경우": name, "취소 확률": round(pp * 100, 1), "위험 등급": u.risk_label(pp)})
            st.dataframe(styled(pd.DataFrame(res)), hide_index=True, use_container_width=True,
                         column_config={"취소 확률": st.column_config.NumberColumn(format="%.1f%%")})
            st.caption("모델이 찾은 관계(상관)이고, 값을 바꾸면 꼭 취소가 줄어든다는 인과는 아니에요.")

        with st.container(key="card_action"):
            st.subheader("권장 대응")
            st.markdown("\n".join(f"1. {t}" for t in u.action_for(risk, agent, deposit)))


# ============================================================
# 4. 취소 위험 대상
# ============================================================
@st.dialog("문자 발송 완료")
def sent_dialog(info):
    n = len(info["예약번호"])
    st.success(f"문자 {n}건 발송이 완료되었어요! (시뮬레이션)")
    first = info["예약번호"][0]
    st.markdown(f"""
| 항목 | 내용 |
| --- | --- |
| 받는 예약 | {first if n == 1 else f"{first} 외 {n - 1}건"} |
| 연락처 | 예약 시스템 연동 시 표시 (데이터에 연락처 없음) |
| 발송 시간 | {pd.Timestamp.now():%Y-%m-%d %H:%M} |
| 메시지 유형 | {info['유형']} |
""")
    if n > 1:
        st.dataframe(pd.DataFrame({"예약번호": [str(x) for x in info["예약번호"]]}),
                     hide_index=True, use_container_width=True, height=180)
    st.caption("실제로 발송되지는 않아요. 실제 운영에서는 예약 시스템의 연락처로 문자 API를 호출해요.")
    if st.button("확인", type="primary", use_container_width=True):
        st.rerun()


def message_box(pool, kind, label, extra=None):
    """한 명씩 고르거나 대상 전체를 골라 → 문구 확인 → 발송(시뮬레이션)"""
    def text_for(r):
        kw = extra(r) if extra else {}
        return u.message(r["hotel"], r["arrival_date"], int(r["nights"]), r["source_row_id"], kind, **kw)

    mode = st.segmented_control("발송 방식", ["한 명씩", f"대상 전체 ({len(pool)}건)"], default="한 명씩",
                                key=f"mode_{kind}") or "한 명씩"
    if mode == "한 명씩":
        i = st.selectbox(label, range(len(pool)),
                         format_func=lambda j: f"{pool.iloc[j]['source_row_id']} · {pool.iloc[j]['arrival_date']:%m-%d} 체크인",
                         key=f"pick_{kind}")
        r = pool.iloc[i]
        st.text_area("문자 내용 (수정 가능)", text_for(r), height=190, key=f"msg_{kind}_{r['source_row_id']}")
        if st.button("문자 발송 (시뮬레이션)", type="primary", key=f"send_{kind}"):
            sent_dialog({"예약번호": [r["source_row_id"]], "유형": kind})
    else:
        st.caption("예약마다 예약번호·체크인 날짜·인원·포인트가 자동으로 들어가요. 아래는 첫 번째 예약의 미리보기예요.")
        st.text_area("문자 미리보기", text_for(pool.iloc[0]), height=190, disabled=True, key=f"preview_{kind}")
        all_msgs = pd.DataFrame({"예약번호": pool["source_row_id"].astype(str),
                                 "체크인 날짜": pool["arrival_date"].dt.strftime("%Y-%m-%d"),
                                 "문자 내용": [text_for(r) for _, r in pool.iterrows()]})
        c1, c2 = st.columns([1, 1])
        if c1.button(f"전체 {len(pool)}건 발송 (시뮬레이션)", type="primary", key=f"send_all_{kind}",
                     use_container_width=True):
            sent_dialog({"예약번호": pool["source_row_id"].tolist(), "유형": kind})
        c2.download_button("전체 문자 내용 CSV 내려받기", all_msgs.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"messages_{kind}_{base_day}.csv", mime="text/csv",
                           key=f"dl_{kind}", use_container_width=True)


def page_contacts():
    header("취소 위험 대상", f"{period} 체크인 예약 · 취소 위험이 높은 예약에 먼저 연락해 예약 유지 여부를 확인해요")
    date_bar()
    blocks, people = u.split_contacts(window, base_day)
    s = u.summary(window)

    c = st.columns(4)
    with c[0].container(key="kpi_hi_block"):
        st.metric("높음 · 단체/여행사", f"{len(blocks)}곳 · {int(blocks['객실'].sum())}건",
                  "한 번 연락으로 여러 객실 확인", delta_color="off")
    with c[1].container(key="kpi_hi_people"):
        st.metric("높음 · 개인 고객", f"{len(people)}건", f"체크인 {u.CONTACT_DAYS}일 전까지 숙박 확인", delta_color="off")
    c[2].metric("보통", f"{s['mid']}건", "리마인드 + 요청사항 입력 안내", delta_color="off")
    c[3].metric("낮음", f"{s['low']}건", "일반 예약 안내만", delta_color="off")

    # ---------- 연락 목록 ----------
    left, right = st.columns(2)
    with left.container(key="card_blocks"):
        st.subheader("① 단체 · 여행사 블록")
        st.caption(f"같은 여행사 · 같은 체크인 날짜 위험 높음 예약이 {u.BLOCK_MIN}건 이상인 단체·오프라인 여행사 예약")
        bt = pd.DataFrame({
            "연락처": "여행사 " + blocks["agent"] + "번",
            "호텔": blocks["hotel"].str.replace(" Hotel", ""),
            "체크인 날짜": blocks["arrival_date"].dt.strftime("%Y-%m-%d"),
            "객실": blocks["객실"],
            "평균 취소 확률": (blocks["평균_취소확률"] * 100).round(1),
            "요청": "명단 확정 요청",
        })
        st.dataframe(bt, hide_index=True, use_container_width=True,
                     column_config={"평균 취소 확률": st.column_config.NumberColumn(format="%.1f%%")})
    with right.container(key="card_people"):
        st.subheader("② 개인 고객 (위험 높음)")
        st.caption(f"연락 기한 = 체크인 {u.CONTACT_DAYS}일 전 · 기한이 빠른 순")
        pt = pd.DataFrame({
            "예약번호": people["source_row_id"].astype(str),
            "체크인 날짜": people["arrival_date"].dt.strftime("%Y-%m-%d"),
            "호텔": people["hotel"].str.replace(" Hotel", ""),
            "취소 확률": (people["prob"] * 100).round(1),
            "연락 기한": [("기한 지남 · 오늘 연락" if late else d.strftime("%m-%d"))
                       for d, late in zip(people["연락 기한"], people["기한 지남"])],
            "주요 요인": people["주요 요인"],
        })
        st.dataframe(pt, hide_index=True, use_container_width=True,
                     column_config={"취소 확률": st.column_config.NumberColumn(format="%.1f%%")})

    contact_csv = pd.concat([bt.assign(구분="단체·여행사"), pt.assign(구분="개인 고객")], ignore_index=True)
    st.download_button("연락 리스트 CSV 내려받기", contact_csv.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"contacts_{base_day}.csv", mime="text/csv")

    with st.container(key="card_msg"):
        st.subheader("예약 확인 안내 문구")
        kind = st.segmented_control("대상", ["개인 · 높음", "보통", "여행사"], default="개인 · 높음",
                                    label_visibility="collapsed") or "개인 · 높음"
        if kind == "여행사":
            if blocks.empty:
                st.info("이 기간에는 단체·여행사 블록이 없어요.")
            else:
                b = blocks.iloc[st.selectbox("여행사", range(len(blocks)),
                                             format_func=lambda i: f"여행사 {blocks.iloc[i]['agent']}번 · "
                                                                   f"{blocks.iloc[i]['arrival_date']:%m-%d} · {blocks.iloc[i]['객실']}건")]
                st.code(u.message(b["hotel"], b["arrival_date"], None, None, "여행사"), language=None, wrap_lines=True)
        else:
            pool = people if kind == "개인 · 높음" else window[window["risk"] == "보통"]
            if pool.empty:
                st.info("이 기간에는 해당하는 예약이 없어요.")
            else:
                message_box(pool, kind, "예약")



# ============================================================
# 5. 프로모션 대상
# ============================================================
def page_promo():
    header("프로모션 대상", f"{period} 체크인 예약 · 취소 위험이 높은 고객에게 체크인 확정 혜택과 보증금 안내 문자를 보내요")
    date_bar()
    blocks, people = u.split_contacts(window, base_day)
    tab2, tab3 = st.tabs(["체크인 확정 리워드", "보증금 안내 문자"])

    # ---------- 탭 1. 체크인 확정 리워드 ----------
    with tab2:
        days_left = (people["arrival_date"] - pd.Timestamp(base_day)).dt.days
        target = people[(days_left >= 0) & (days_left <= u.REWARD_DAYS)].copy()
        target["남은 일수"] = days_left[target.index]
        target["웰컴 드링크"] = (target["adults"] + target["children"]).astype(int)
        target["포인트"] = target["lead_time"].apply(u.reward_points)   # 처음 예약할 때의 예약~체크인 기간 기준

        with st.container(key="card_reward_rule"):
            st.subheader("체크인 확정 리워드")
            st.markdown(
                f"- **발송**: 체크인 **{u.REWARD_DAYS}일 전**, 위험 높음 개인 고객에게 안내 문자\n"
                f"- **혜택**: 체크인을 확정하면 무료 웰컴 드링크 **1인 1잔** + 포인트 적립 (1포인트 = 1원, 체크인 시 지급)\n"
                f"- **포인트**: 예약~체크인 기간 **{u.POINT_FULL_DAYS}일 이상 → {u.POINT_FULL:,}포인트** "
                f"(취소가 특히 많은 구간이라 높게) · {u.POINT_FULL_DAYS}일 미만 → **기간 × {u.POINT_PER_DAY}포인트**")
            p1, p2 = st.columns([1, 2], vertical_alignment="bottom")
            d = p1.number_input("포인트 계산기: 예약~체크인 기간 (일)", 0, 800, 200, step=10)
            p2.metric("확정 시 지급 포인트", f"{u.reward_points(d):,}P", f"= {u.reward_points(d):,}원", delta_color="off")

        if target.empty:
            st.info(f"이 기간에는 체크인까지 {u.REWARD_DAYS}일 이내인 위험 높음 개인 예약이 없어요. 기준일을 바꿔 보세요.")
        else:
            with st.container(key="card_reward_list"):
                st.subheader(f"리워드 안내 대상 {len(target)}건")
                rt = pd.DataFrame({
                    "예약번호": target["source_row_id"].astype(str),
                    "체크인": target["arrival_date"].dt.strftime("%Y-%m-%d"),
                    "남은 일수": target["남은 일수"],
                    "예약~체크인 기간": target["lead_time"],
                    "호텔": target["hotel"].str.replace(" Hotel", ""),
                    "취소 확률": (target["prob"] * 100).round(1),
                    "웰컴 드링크(잔)": target["웰컴 드링크"],
                    "확정 시 포인트": target["포인트"],
                })
                st.dataframe(rt, hide_index=True, use_container_width=True,
                             column_config={"취소 확률": st.column_config.NumberColumn(format="%.1f%%"),
                                            "확정 시 포인트": st.column_config.NumberColumn(format="%d P")})
            with st.container(key="card_reward_msg"):
                st.subheader("리워드 안내 문자")
                message_box(target, "체크인 확정 리워드", "예약",
                            extra=lambda r: {"guests": int(r["웰컴 드링크"]), "points": int(r["포인트"])})
                st.caption("혜택이 들어간 광고성 문자라 (광고) 표시와 수신거부 안내를 넣었어요. 수신 동의한 고객에게만 보내야 해요.")

    # ---------- 탭 2. 보증금 안내 문자 ----------
    with tab3:
        dep = window[(window["risk"] == "높음") & (window["deposit_type"] == "No Deposit")].sort_values("arrival_date")
        with st.container(key="card_deposit_rule"):
            st.subheader(f"예약 보증금 {u.DEPOSIT_KRW:,}원 안내")
            st.markdown(f"- **대상**: 위험 높음 예약 중 **보증금이 없는** 예약 ({len(dep)}건)\n"
                        f"- **내용**: 예약 확정을 위해 보증금 {u.DEPOSIT_KRW:,}원 납부 안내 → 입금 확인 후 최종 확정")
            st.caption("이미 확정된 예약에 보증금을 새로 요청하는 방식이라, 실제 운영 시에는 예약 약관 확인이 필요해요.")
        if dep.empty:
            st.info("이 기간에는 보증금 안내 대상이 없어요.")
        else:
            with st.container(key="card_deposit_list"):
                dt = pd.DataFrame({
                    "예약번호": dep["source_row_id"].astype(str),
                    "체크인": dep["arrival_date"].dt.strftime("%Y-%m-%d"),
                    "호텔": dep["hotel"].str.replace(" Hotel", ""),
                    "취소 확률": (dep["prob"] * 100).round(1),
                    "1박 요금": dep["adr"].apply(u.won),
                    "현재 보증금": "없음",
                })
                st.dataframe(dt, hide_index=True, use_container_width=True, height=260,
                             column_config={"취소 확률": st.column_config.NumberColumn(format="%.1f%%")})
            with st.container(key="card_deposit_msg"):
                st.subheader("보증금 안내 문자")
                message_box(dep, "보증금 안내", "예약")


{"홈": page_home, "오늘 현황": page_today, "예약 조회": page_bookings,
 "새 예약 예측": page_predict, "취소 위험 대상": page_contacts,
 "프로모션 대상": page_promo}[st.session_state.page]()
