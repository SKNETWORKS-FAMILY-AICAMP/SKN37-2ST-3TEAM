"""
호텔 예약 취소 위험 관리 앱 - 계산 로직 모음
(화면 코드는 app.py, 여기는 Streamlit 없이도 돌아가는 계산만)

어떤 모델이든 사용할 수 있게 만들었어요.
조건: 전처리까지 포함된 sklearn Pipeline을 joblib으로 저장한 파일
      (DataFrame으로 학습해서 model.feature_names_in_ 가 있어야 함)
"""
from datetime import date, timedelta

import struct

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_stacking_modelB2.joblib"
DATA_PATH = BASE_DIR / "data" / "hotel_preprocessed.csv"
# ------------------------------------------------------------
# 설정: 최종 모델이 정해지면 여기만 바꾸면 돼요
# ------------------------------------------------------------
#MODEL_PATH = "best_stacking_modelB2.joblib"   # 최종 선정 모델 (스태킹 B, 연도 제외) · TeamProject2 폴더에 둠
#DATA_PATH = "hotel_preprocessed.csv"     # 공통 전처리 결과
MODEL_LABEL = "Stacking B"   # 화면에 보여줄 모델 이름 (팀 최종 선정 모델)
THRESHOLD = 0.50      # 이 확률 이상이면 "취소 예측" — 스태킹 B 담당자가 정한 임계값으로 바꿀 것
HIGH = 0.70           # 이 확률 이상이면 위험 "높음"
BLOCK_MIN = 5         # 같은 여행사 · 같은 체크인 날짜 위험 예약이 이 수 이상이면 "단체 블록"
BLOCK_SEGMENTS = ["Groups", "Offline TA/TO"]   # 단체 블록으로 볼 예약 경로 (온라인 개인 예약은 제외)
MAX_LEAD_DAYS = 730   # 예약~체크인 기간 최대 2년 (학습 데이터 최대 737일)
CONTACT_DAYS = 7      # 개인 고객 연락 기한: 체크인 N일 전

# 금액: 데이터는 포르투갈 호텔이라 유로(€) → 화면에는 원화로 바꿔서 보여줌
EUR_KRW = 1480        # 1유로 = 1,480원 (2026-10 환율 기준 고정값, 필요하면 여기만 수정)

# 프로모션 규칙 (팀 회의 결정)
REWARD_DAYS = 14          # 체크인 14일 전에 체크인 확정 리워드 안내 문자 발송
# 포인트는 처음 예약할 때의 예약~체크인 기간(lead_time) 기준
#   180일 이상: 취소가 특히 많아서 30,000포인트로 높게
#   180일 미만: 기간에 비례해서 lead_time × 100포인트 (1포인트 = 1원)
POINT_FULL = 30000
POINT_FULL_DAYS = 180
POINT_PER_DAY = 100
DEPOSIT_KRW = 100000      # 예약 보증금 안내 금액

TARGET = "is_canceled"
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
DATA_START, DATA_END = date(2015, 7, 1), date(2017, 8, 31)   # 학습 데이터 기간

# 주요 요인으로 보여줄 컬럼과 한국어 이름
REASON_COLS = ["country", "agent", "deposit_type", "lead_time",
               "total_of_special_requests", "previous_cancellations",
               "previous_bookings_not_canceled", "market_segment",
               "customer_type", "is_repeated_guest", "adr"]


def won(eur):
    """유로 금액 → '123,000원' 문자열"""
    return f"{round(eur * EUR_KRW, -2):,.0f}원"


def won_short(eur):
    """큰 금액을 '1.41억 원' / '9,500만 원' 처럼 짧게"""
    krw = eur * EUR_KRW
    return f"{krw / 1e8:.2f}억 원" if krw >= 1e8 else f"{krw / 1e4:,.0f}만 원"


def reward_points(lead_time):
    """체크인 확정 리워드 포인트: 예약~체크인 기간 180일 이상이면 30,000, 아니면 기간 × 100"""
    lead_time = max(int(lead_time), 0)
    return POINT_FULL if lead_time >= POINT_FULL_DAYS else lead_time * POINT_PER_DAY


# ------------------------------------------------------------
# 1. 불러오기
# ------------------------------------------------------------
class StackingModel:
    """
    스태킹 모델 파일이 {"base_models": {이름: Pipeline}, "meta_model": 모델} 딕셔너리로 저장돼 있을 때
    하나의 모델처럼 쓰게 감싸는 클래스
      1) 베이스 모델(XGBoost · Random Forest · MLP)이 각각 취소 확률을 계산
      2) 메타 모델이 그 확률 3개를 받아 최종 취소 확률을 계산
    """
    def __init__(self, bundle):
        self.base_models = bundle["base_models"]
        self.meta_model = bundle["meta_model"]
        first = next(iter(self.base_models.values()))
        self.feature_names_in_ = np.array([str(c) for c in first.feature_names_in_], dtype=object)   # 앱 입력 컬럼
        self.meta_cols = list(getattr(self.meta_model, "feature_names_in_", self.base_models.keys()))

    def predict_proba(self, X):
        base = pd.DataFrame({name: m.predict_proba(X[[str(c) for c in m.feature_names_in_]])[:, 1]
                             for name, m in self.base_models.items()}, index=X.index)
        return self.meta_model.predict_proba(base[self.meta_cols])


# ------------------------------------------------------------
# XGBoost 모델 복원 문제 대비
#   pickle로 저장된 XGBoost가 "input stream corrupted"로 안 열릴 때,
#   저장된 데이터에서 모델 본체("Model")만 꺼내 load_model로 다시 읽음
# ------------------------------------------------------------
def _ubj_model_slice(raw):
    """XGBoost 직렬화 데이터({"Config":..., "Model":...})에서 "Model" 부분 바이트만 잘라냄"""
    raw = bytes(raw); pos = [0]
    def rd(n):
        b = raw[pos[0]:pos[0] + n]
        if len(b) < n: raise ValueError("unexpected end")
        pos[0] += n; return b
    def num(t):
        f = {b'i': ('>b', 1), b'U': ('>B', 1), b'I': ('>h', 2), b'l': ('>i', 4), b'L': ('>q', 8), b'd': ('>f', 4), b'D': ('>d', 8)}[t]
        return struct.unpack(f[0], rd(f[1]))[0]
    def skip(t=None):
        if t is None: t = rd(1)
        if t in b'iUIlLdD': num(t); return
        if t == b'S': rd(num(rd(1))); return
        if t in b'TFZ': return
        if t == b'{':
            while raw[pos[0]:pos[0] + 1] != b'}': rd(num(rd(1))); skip()
            rd(1); return
        if t == b'[':
            nxt = raw[pos[0]:pos[0] + 1]
            if nxt == b'$':
                rd(1); et = rd(1); rd(1); n = num(rd(1))
                rd(n * {b'i': 1, b'U': 1, b'I': 2, b'l': 4, b'L': 8, b'd': 4, b'D': 8}[et]); return
            if nxt == b'#':
                rd(1)
                for _ in range(num(rd(1))): skip()
                return
            while raw[pos[0]:pos[0] + 1] != b']': skip()
            rd(1); return
        raise ValueError(f"bad type {t!r}")
    assert rd(1) == b'{'
    while raw[pos[0]:pos[0] + 1] != b'}':
        key = rd(num(rd(1))).decode()
        start = pos[0]; skip()
        if key == "Model": return raw[start:pos[0]]
    raise ValueError("Model not found")


def _joblib_load(path):
    try:
        return joblib.load(path)
    except Exception as e:
        if "xgboost" not in type(e).__module__.lower():
            raise
    import xgboost as xgb
    original = xgb.Booster.__setstate__

    def safe_setstate(self, state):
        try:
            original(self, dict(state))
        except Exception:
            tmp = xgb.Booster()
            tmp.load_model(bytearray(_ubj_model_slice(state["handle"])))
            new_state = dict(state)
            new_state["handle"] = tmp.handle
            tmp.handle = None          # tmp가 지워질 때 모델이 같이 지워지지 않게
            self.__dict__.update(new_state)

    xgb.Booster.__setstate__ = safe_setstate
    try:
        return joblib.load(path)
    finally:
        xgb.Booster.__setstate__ = original


def load_model(path=MODEL_PATH):
    obj = _joblib_load(path)
    if isinstance(obj, dict) and "base_models" in obj:   # 스태킹 딕셔너리 → 감싸기
        return StackingModel(obj)
    return obj


def model_features(model):
    """모델이 학습할 때 받은 입력 컬럼 목록 (year를 뺐는지도 여기서 자동으로 반영)"""
    if not hasattr(model, "feature_names_in_"):
        raise ValueError("모델을 DataFrame으로 학습해야 해요 (feature_names_in_ 없음)")
    return [str(c) for c in model.feature_names_in_]


def model_name(model):
    if isinstance(model, StackingModel):
        return "Stacking (" + " + ".join(model.base_models) + ")"
    last = model.steps[-1][1] if hasattr(model, "steps") else model
    return type(last).__name__


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    for c in df.columns:                       # 범주형은 문자로 통일 (agent "9" 등)
        if df[c].dtype == object or str(df[c].dtype).startswith("str"):
            df[c] = df[c].astype(str)
    month_num = df["arrival_date_month"].map({m: i for i, m in enumerate(MONTHS, 1)})
    df["arrival_date"] = pd.to_datetime(dict(year=df["arrival_date_year"],
                                             month=month_num,
                                             day=df["arrival_date_day_of_month"]))
    df["nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    return df


# ------------------------------------------------------------
# 2. 예측과 위험 등급
# ------------------------------------------------------------
def predict(model, X):
    return model.predict_proba(X[model_features(model)])[:, 1]


def risk_label(p):
    if p >= HIGH:
        return "높음"
    if p >= THRESHOLD:
        return "보통"
    return "낮음"


def add_predictions(model, df):
    out = df.copy()
    out["prob"] = predict(model, out)
    out["risk"] = out["prob"].apply(risk_label)
    return out


def baseline_values(train):
    """'보통 예약' 값: 범주형은 가장 많은 값, 수치형은 중앙값"""
    base = {}
    for c in train.columns:
        if train[c].dtype == object or str(train[c].dtype).startswith("str"):
            base[c] = train[c].mode()[0]
        elif np.issubdtype(train[c].dtype, np.number):
            base[c] = train[c].median()
    return base


# ------------------------------------------------------------
# 3. 주요 요인 (모델 종류와 상관없이 쓰는 방법)
#    한 컬럼 값을 학습 데이터의 다른 예약 값들로 바꿔 가며 다시 예측하고,
#    원래 값일 때와 평균적으로 얼마나 차이 나는지 봄
#    + : 이 값 때문에 취소 가능성이 올라감(↑)  /  - : 내려감(↓)
# ------------------------------------------------------------
def background_sample(train, n=20, seed=0):
    """비교 기준으로 쓸 학습 데이터 예약 n건"""
    return train.sample(n=n, random_state=seed).reset_index(drop=True)


def _logit(p):
    # 확률 대신 log-odds로 비교: 99%처럼 끝에 몰린 확률에서도 요인 차이가 보이게
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def explain(model, rows, background):
    feats = model_features(model)
    cols = [c for c in REASON_COLS if c in feats]
    rows = rows[feats]
    p0 = _logit(predict(model, rows))
    # 바꾼 예약들을 전부 한 표로 모아 예측을 '한 번'만 실행 (무거운 모델도 빠르게)
    batch = []
    for c in cols:
        for v in background[c]:
            changed = rows.copy()
            changed[c] = v
            batch.append(changed)
    p = _logit(predict(model, pd.concat(batch, ignore_index=True)))
    p = p.reshape(len(cols), len(background), len(rows))   # (컬럼, 비교 예약, 원래 예약)
    effect = pd.DataFrame({c: p0 - p[i].mean(axis=0) for i, c in enumerate(cols)}, index=rows.index)
    return effect


def phrase(col, v):
    if col == "country":
        return "국내 고객(포르투갈)" if v == "PRT" else f"국적({v})"
    if col == "agent":
        return "여행사 없음(직접 예약)" if v == "Unknown" else f"여행사 {v}번"
    if col == "deposit_type":
        return {"Non Refund": "환불 불가 예약", "No Deposit": "보증금 없음",
                "Refundable": "환불 가능 보증금"}.get(v, v)
    if col == "lead_time":
        return f"예약~체크인 {int(v)}일"
    if col == "total_of_special_requests":
        return "특별 요청 없음" if v == 0 else f"특별 요청 {int(v)}개"
    if col == "previous_cancellations":
        return f"이전 취소 {int(v)}회"
    if col == "previous_bookings_not_canceled":
        return f"이전 이용 {int(v)}회"
    if col == "market_segment":
        return f"예약 경로 {v}"
    if col == "customer_type":
        return f"고객 유형 {v}"
    if col == "is_repeated_guest":
        return "재방문 고객" if v == 1 else "첫 방문 고객"
    if col == "adr":
        return f"1박 요금 {won(v)}"
    return f"{col}={v}"


def reason_list(row, effect_row, top=3, min_effect=0.15):
    """[(방향, 문구, 효과)] - 효과가 큰 순서"""
    items = [(c, e) for c, e in effect_row.items() if abs(e) >= min_effect]
    items.sort(key=lambda x: -abs(x[1]))
    return [("↑" if e > 0 else "↓", phrase(c, row[c]), e) for c, e in items[:top]]


def reason_text(row, effect_row, top=2):
    r = reason_list(row, effect_row, top=top)
    return " · ".join(f"{d} {t}" for d, t, _ in r) if r else "뚜렷한 요인 없음"


# ------------------------------------------------------------
# 4. 기준일 기준 체크인 예정 예약
# ------------------------------------------------------------
def upcoming(pred_df, base_day, days):
    start = pd.Timestamp(base_day)
    end = start + pd.Timedelta(days=days)
    return pred_df[(pred_df["arrival_date"] >= start) & (pred_df["arrival_date"] < end)].copy()


def summary(window):
    return {
        "total": len(window),
        "high": int((window["risk"] == "높음").sum()),
        "mid": int((window["risk"] == "보통").sum()),
        "low": int((window["risk"] == "낮음").sum()),
        "expected_cancel": float(window["prob"].sum()),
        "revenue_at_risk": float((window["prob"] * window["adr"] * window["nights"]).sum()),
    }


def split_contacts(window, base_day):
    """위험 높음 예약을 ① 단체·여행사 블록 ② 개인 고객으로 나눔"""
    hi = window[window["risk"] == "높음"].copy()
    key = ["hotel", "agent", "arrival_date"]
    size = hi.groupby(key)["prob"].transform("size")
    is_block = ((hi["agent"] != "Unknown") & (size >= BLOCK_MIN)
                & hi["market_segment"].isin(BLOCK_SEGMENTS))

    blocks = (hi[is_block].groupby(key)
              .agg(객실=("prob", "size"), 평균_취소확률=("prob", "mean"),
                   예약경로=("market_segment", lambda s: s.mode()[0]),
                   보증금=("deposit_type", lambda s: s.mode()[0]))
              .reset_index().sort_values(["arrival_date", "객실"], ascending=[True, False]))

    people = hi[~is_block].copy()
    deadline = people["arrival_date"] - pd.Timedelta(days=CONTACT_DAYS)
    people["연락 기한"] = deadline
    people["기한 지남"] = deadline < pd.Timestamp(base_day)
    people = people.sort_values(["연락 기한", "prob"], ascending=[True, False])
    return blocks, people


# ------------------------------------------------------------
# 5. 새 예약 입력 → 모델 입력 한 줄 만들기
# ------------------------------------------------------------
def build_input(form, feats, base):
    """form: 화면에서 받은 값(dict). 모델이 쓰는 컬럼 중 화면에 없는 값은 '보통 예약' 값으로 채움"""
    arrival, booked = form["arrival_date"], form["booking_date"]
    row = dict(form)
    row.pop("arrival_date"), row.pop("booking_date")
    row.update({
        "lead_time": (arrival - booked).days,
        "arrival_date_year": arrival.year,
        "arrival_date_month": MONTHS[arrival.month - 1],
        "arrival_date_week_number": int(arrival.isocalendar()[1]),
        "arrival_date_day_of_month": arrival.day,
        "total_guests": row["adults"] + row["children"] + row["babies"],
        "total_nights": row["stays_in_weekend_nights"] + row["stays_in_week_nights"],
    })
    full = {c: row.get(c, base.get(c)) for c in feats}
    return pd.DataFrame([full])


def form_from_row(r):
    """기존 예약 한 줄 → 새 예약 예측 화면의 기본값"""
    arrival = r["arrival_date"].date()
    return {
        "hotel": r["hotel"], "arrival_date": arrival,
        "booking_date": arrival - timedelta(days=int(r["lead_time"])),
        "stays_in_weekend_nights": int(r["stays_in_weekend_nights"]),
        "stays_in_week_nights": int(r["stays_in_week_nights"]),
        "adults": int(r["adults"]), "children": int(r["children"]), "babies": int(r["babies"]),
        "reserved_room_type": r["reserved_room_type"], "meal": r["meal"], "adr": float(r["adr"]),
        "total_of_special_requests": int(r["total_of_special_requests"]),
        "deposit_type": r["deposit_type"], "market_segment": r["market_segment"],
        "distribution_channel": r["distribution_channel"], "agent": r["agent"],
        "customer_type": r["customer_type"], "country": r["country"],
        "is_repeated_guest": int(r["is_repeated_guest"]),
        "previous_cancellations": int(r["previous_cancellations"]),
        "previous_bookings_not_canceled": int(r["previous_bookings_not_canceled"]),
        "days_in_waiting_list": int(r["days_in_waiting_list"]),
        "has_company": int(r["has_company"]),
    }


def action_for(risk, agent, deposit):
    if risk == "높음":
        steps = [f"체크인 {CONTACT_DAYS}일 전까지 숙박 확인 안내 발송",
                 "객실·식사 요청사항 입력 안내 함께 보내기"]
        if deposit == "Non Refund" and agent != "Unknown":
            steps = [f"여행사 {agent}번에 단체 명단(인원) 확정 요청"] + steps[:1]
        elif agent != "Unknown":
            steps.append(f"회신이 없으면 여행사 {agent}번에 예약 유지 확인")
        return steps
    if risk == "보통":
        return ["체크인 전 리마인드 안내", "객실·식사 요청사항 입력 안내"]
    return ["일반 예약 안내만"]


def message(hotel, arrival, nights, booking_no, kind="개인 · 높음", guests=1, points=0):
    a = pd.Timestamp(arrival).strftime("%Y-%m-%d")
    if kind == "체크인 확정 리워드":
        return (f"(광고) [{hotel}] 체크인 확정 리워드 안내\n"
                f"안녕하세요. {a} 체크인 예정인 예약(예약번호 {booking_no})을 확인드립니다.\n"
                f"체크인을 확정하시면 포인트와 웰컴 드링크를 받을 수 있어요.\n"
                f"· 무료 웰컴 드링크 {guests}잔 (1인 1잔)\n"
                f"· {points:,}포인트 적립 (1포인트 = 1원, 체크인 시 지급)\n"
                f"무료 수신거부 080-000-0000")
    if kind == "보증금 안내":
        return (f"[{hotel}] 예약 보증금 안내\n"
                f"안녕하세요. 고객님의 예약(예약번호 {booking_no}, {a} 체크인)이 확인되었습니다.\n"
                f"예약 확정을 위해 보증금 {DEPOSIT_KRW:,}원을 납부해 주시기 바랍니다.\n"
                f"입금 확인 후 예약이 최종 확정됩니다. 감사합니다.")
    if kind == "여행사":
        return (f"[{hotel}] 단체 예약 인원 확정 요청\n"
                f"안녕하세요. {a} 체크인 단체 예약의 최종 인원과 투숙객 명단을 확인 부탁드립니다.\n"
                f"변동이 있으면 체크인 {CONTACT_DAYS}일 전까지 알려 주시면 객실을 맞춰 준비하겠습니다.")
    if kind == "보통":
        return (f"[{hotel}] 체크인 안내\n"
                f"안녕하세요. {a} 체크인, {nights}박 예약(예약번호 {booking_no})을 기다리고 있어요.\n"
                f"객실·식사 관련 요청사항이 있으면 미리 남겨 주세요.")
    return (f"[{hotel}] 예약 확인 안내\n"
            f"안녕하세요. {a} 체크인, {nights}박 예약(예약번호 {booking_no})을 확인드립니다.\n"
            f"예정대로 숙박하시면 회신으로 숙박 확인을 부탁드려요. "
            f"객실·식사 관련 요청사항이 있으면 함께 남겨 주세요.\n"
            f"일정이 바뀌었다면 미리 알려 주시면 감사하겠습니다.")
