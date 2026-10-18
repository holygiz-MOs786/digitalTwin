"""
Digital Twin Dashboard - AI for Predictive Maintenance in Manufacturing
Replays machine sensor data as a live feed and predicts failure risk with XGBoost.
Files needed in the same folder: pdm_model.pkl, machine_data.csv
"""
import time
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Machine Digital Twin", page_icon="⚙️", layout="wide")

FEATURES = ['air_temp', 'process_temp', 'speed', 'torque', 'tool_wear', 'temp_diff', 'power']
WINDOW = 60  # readings shown in trend charts


# ---------- Load model + data (cached, runs once) ----------
@st.cache_resource
def load_model():
    return joblib.load("pdm_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("machine_data.csv")


bundle = load_model()
model, scaler, cols = bundle["model"], bundle["scaler"], bundle["columns"]
tuned_t = float(bundle.get("threshold", 0.5))


def prepare(rows: pd.DataFrame) -> np.ndarray:
    """Same feature steps as training: engineered features + one-hot Type + scaling."""
    rows = rows.copy()
    rows["temp_diff"] = rows["process_temp"] - rows["air_temp"]
    rows["power"] = rows["torque"] * rows["speed"] * 2 * np.pi / 60
    X = pd.get_dummies(rows[FEATURES + ["Type"]], columns=["Type"], dtype=int)
    X = X.reindex(columns=cols, fill_value=0)
    return scaler.transform(X)


@st.cache_data
def all_risks(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


data = load_data()
data["risk"] = all_risks(data)

# ---------- Session state ----------
ss = st.session_state
ss.setdefault("idx", WINDOW)
ss.setdefault("running", False)
ss.setdefault("alerts", [])

# ---------- Sidebar controls ----------
st.sidebar.title("⚙️ Twin Controls")
c1, c2 = st.sidebar.columns(2)
if c1.button("▶ Start", width="stretch"):
    ss.running = True
if c2.button("⏸ Pause", width="stretch"):
    ss.running = False
if st.sidebar.button("⟲ Reset", width="stretch"):
    ss.idx, ss.running, ss.alerts = WINDOW, False, []

speed_s = st.sidebar.slider("Seconds per reading", 0.2, 3.0, 1.0, 0.1)
threshold = st.sidebar.slider("Alarm threshold", 0.05, 0.95, round(tuned_t, 2), 0.01,
                              help=f"Tuned value = {tuned_t:.2f}. Lower = catch more failures, more false alarms.")
jump = st.sidebar.number_input("Jump to reading #", WINDOW, len(data) - 1, ss.idx, step=1)
if st.sidebar.button("Go", width="stretch"):
    ss.idx = int(jump)
fail_idx = data.index[data["Machine failure"] == 1]
st.sidebar.caption("Demo tip - real failures occur at readings: "
                   + ", ".join(str(x) for x in fail_idx[fail_idx >= WINDOW][:8]) + " ...")

# ---------- Current state ----------
i = min(ss.idx, len(data) - 1)
row = data.iloc[i]
hist = data.iloc[max(0, i - WINDOW):i + 1]
risk = float(row["risk"])

if risk >= threshold:
    status, color = "CRITICAL - maintenance required", "#d62728"
elif risk >= threshold / 2:
    status, color = "WARNING - monitor closely", "#ff9f1c"
else:
    status, color = "HEALTHY", "#2ca02c"

if risk >= threshold and (not ss.alerts or ss.alerts[-1]["Reading"] != i):
    ss.alerts.append({"Reading": i, "Product": row["Product ID"], "Risk %": round(risk * 100, 1),
                      "Torque [Nm]": row["torque"], "Tool wear [min]": int(row["tool_wear"]),
                      "Actual failure": "Yes" if row["Machine failure"] == 1 else "No"})

# ---------- Header ----------
st.title("Machine Digital Twin - Predictive Maintenance")
st.markdown(
    f"<div style='padding:12px 18px;border-radius:10px;background:{color};color:white;"
    f"font-size:20px;font-weight:600'>Status: {status} &nbsp;|&nbsp; Reading #{i} &nbsp;|&nbsp; "
    f"Product {row['Product ID']} (Type {row['Type']})</div>", unsafe_allow_html=True)
st.write("")

tab_live, tab_whatif = st.tabs(["📡 Live Monitor", "🧪 What-if Simulator"])

with tab_live:
    k = st.columns(6)
    prev = data.iloc[i - 1]
    k[0].metric("Air Temp (K)", f"{row['air_temp']:.1f}", f"{row['air_temp'] - prev['air_temp']:+.1f}")
    k[1].metric("Process Temp (K)", f"{row['process_temp']:.1f}", f"{row['process_temp'] - prev['process_temp']:+.1f}")
    k[2].metric("Speed (rpm)", f"{row['speed']:.0f}", f"{row['speed'] - prev['speed']:+.0f}")
    k[3].metric("Torque (Nm)", f"{row['torque']:.1f}", f"{row['torque'] - prev['torque']:+.1f}")
    k[4].metric("Tool Wear (min)", f"{row['tool_wear']:.0f}")
    k[5].metric("Power (W)", f"{row['power']:.0f}")

    g, r = st.columns([1, 2])
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=risk * 100, number={"suffix": "%"},
        title={"text": "Failure Risk"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [{"range": [0, threshold * 50], "color": "#e8f5e9"},
                         {"range": [threshold * 50, threshold * 100], "color": "#fff3e0"},
                         {"range": [threshold * 100, 100], "color": "#ffebee"}],
               "threshold": {"line": {"color": "black", "width": 3}, "value": threshold * 100}}))
    gauge.update_layout(height=300, margin=dict(t=50, b=10, l=20, r=20))
    g.plotly_chart(gauge, width="stretch")

    rf = go.Figure()
    rf.add_trace(go.Scatter(x=hist.index, y=hist["risk"] * 100, mode="lines", name="Risk %",
                            line=dict(color="#d62728", width=2), fill="tozeroy"))
    rf.add_hline(y=threshold * 100, line_dash="dash", annotation_text="Alarm threshold")
    fails = hist[hist["Machine failure"] == 1]
    rf.add_trace(go.Scatter(x=fails.index, y=fails["risk"] * 100, mode="markers", name="Actual failure",
                            marker=dict(symbol="x", size=12, color="black")))
    rf.update_layout(title="Risk Trend", height=300, yaxis_title="Risk %", yaxis_range=[0, 100],
                     margin=dict(t=50, b=10, l=20, r=20), legend=dict(orientation="h", y=-0.2))
    r.plotly_chart(rf, width="stretch")

    s1, s2 = st.columns(2)
    for col, (a, b, title) in zip([s1, s2], [("torque", "speed", "Torque vs Speed"),
                                             ("air_temp", "process_temp", "Temperatures")]):
        f = go.Figure()
        f.add_trace(go.Scatter(x=hist.index, y=hist[a], name=a, line=dict(width=2)))
        f.add_trace(go.Scatter(x=hist.index, y=hist[b], name=b, yaxis="y2", line=dict(width=2)))
        f.update_layout(title=title, height=280, margin=dict(t=50, b=10, l=20, r=20),
                        yaxis=dict(title=a), yaxis2=dict(title=b, overlaying="y", side="right"),
                        legend=dict(orientation="h", y=-0.25))
        col.plotly_chart(f, width="stretch")

    st.subheader(f"🚨 Alert Log ({len(ss.alerts)})")
    if ss.alerts:
        st.dataframe(pd.DataFrame(ss.alerts[::-1]), width="stretch", hide_index=True)
    else:
        st.caption("No alerts yet.")

with tab_whatif:
    st.markdown("Change the machine's operating conditions and see the predicted risk instantly.")
    w1, w2 = st.columns(2)
    t_type = w1.selectbox("Product type", ["L", "M", "H"])
    air = w1.slider("Air temperature (K)", 295.0, 305.0, 300.0, 0.1)
    proc = w1.slider("Process temperature (K)", 305.0, 314.0, 310.0, 0.1)
    rpm = w2.slider("Rotational speed (rpm)", 1150, 2900, 1500, 10)
    tq = w2.slider("Torque (Nm)", 3.0, 77.0, 40.0, 0.5)
    wear = w2.slider("Tool wear (min)", 0, 255, 100, 1)
    sim = pd.DataFrame([{"air_temp": air, "process_temp": proc, "speed": rpm,
                         "torque": tq, "tool_wear": wear, "Type": t_type}])
    p = float(model.predict_proba(prepare(sim))[0, 1])
    verdict = "FAILURE LIKELY" if p >= threshold else ("AT RISK" if p >= threshold / 2 else "SAFE")
    st.metric("Predicted failure risk", f"{p * 100:.1f}%", verdict, delta_color="off")
    st.progress(min(p, 1.0))
    pw = tq * rpm * 2 * np.pi / 60
    st.caption(f"Derived: temperature difference = {proc - air:.1f} K, power = {pw:.0f} W. "
               "Try: tool wear > 200 with torque > 50, or speed < 1380 with temp difference < 8.6.")

# ---------- Advance the live feed ----------
if ss.running and ss.idx < len(data) - 1:
    time.sleep(speed_s)
    ss.idx += 1
    st.rerun()
