# AnkaGrid - Orman Şefliği Karar Destek Paneli (SİMÜLASYON) | tek dosya: app.py
# Yerelde:  python app.py   (eksik kütüphaneler otomatik kurulur ve uygulama açılır)
# Bulutta:  requirements.txt kullanılır (bu blok orada zaten atlanır)
import importlib.util, subprocess, sys, os

def _ensure(pkgs):
    missing = [p for p in pkgs if importlib.util.find_spec(p) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

_ensure(["streamlit", "pandas", "numpy"])

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    _in_streamlit = get_script_run_ctx(suppress_warning=True) is not None
except Exception:
    _in_streamlit = False
if not _in_streamlit and __name__ == "__main__":      # "python app.py" ile çalıştırıldıysa
    subprocess.call([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])
    sys.exit()

import math
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AnkaGrid", page_icon="🌲", layout="wide")

# ===================== 1) AĞAÇ PROFİLİ (DİNAMİK) =====================
@dataclass(frozen=True)
class TreeProfile:
    species: str
    critical_temp: float     # X: kritik gövde ısısı (°C)
    critical_moist: float    # Y: kritik iç nem (%)
    drought_moist: float     # kuraklık stresi eşiği (%)
    max_rate: float          # Z: ΔT/Δt eşiği (°C/dk)
    max_excess: float        # gövde-ortam farkı üst sınırı (°C)
    max_moist_drop: float    # nem düşüş hızı eşiği (%/dk)
    confirm: int             # ardışık doğrulama sayısı

PROFILES = {
    "Kızılçam": TreeProfile("Pinus brutia", 58, 30, 45, 3.0, 12, 4.0, 3),
    "Karaçam":  TreeProfile("Pinus nigra",  60, 32, 47, 3.0, 12, 4.0, 3),
}

NORMAL, DROUGHT, WATCH, FIRE = 0, 1, 2, 3
STATES = {NORMAL: ("NORMAL", "#2ecc71"), DROUGHT: ("KURAKLIK STRESİ", "#f1c40f"),
          WATCH: ("İZLEME", "#e67e22"), FIRE: ("YANGIN ALARMI", "#e74c3c")}
NODE_LAT, NODE_LON = 36.8841, 30.7056

# ===================== 2) SENTETİK SENSÖR VERİSİ =====================
def simulate(scn, n=120, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    amb = 32 + 1.2 * np.sin(t / 25) + rng.normal(0, .03, n)
    if scn == "Normal gün":
        trunk, moist = amb + 1.5, 55 - .01 * t
    elif scn == "Kuraklık stresi":
        trunk, moist = amb + 2.0, 43 - .03 * t
    else:  # kuru ağaçta yangın başlangıcı (dk 70'te)
        k = np.clip(t - 70, 0, None)
        trunk = amb + 1.5 + np.minimum(3.4 * k, 32)
        moist = 40 - .02 * t - np.minimum(4.5 * k, 28)
    trunk = trunk + rng.normal(0, .05, n)
    moist = moist + rng.normal(0, .3, n)
    return pd.DataFrame({"dk": t, "Gövde °C": trunk, "Ortam °C": amb, "İç nem %": moist})

# ===================== 3) ANOMALİ TESPİTİ (cihaz kodunun aynısı) =====================
def evaluate(df, p):
    T, A, H = (df[c].to_numpy() for c in ("Gövde °C", "Ortam °C", "İç nem %"))
    n = len(df)
    rate, drop = np.r_[0, np.diff(T)], np.r_[0, -np.diff(H)]
    states, scores, sus = np.zeros(n, int), np.zeros(n, int), 0
    for i in range(n):
        ex = T[i] - A[i]
        s = (int(T[i] >= p.critical_temp) + int(rate[i] >= p.max_rate) + int(ex >= p.max_excess)
             + int(drop[i] >= p.max_moist_drop) + int(H[i] <= p.critical_moist))
        early = rate[i] >= p.max_rate * .2 or ex >= p.max_excess * .5
        if s >= 3:
            sus += 1
            state = FIRE if sus >= p.confirm else WATCH
        elif early or s == 2:
            sus, state = 0, WATCH
        else:
            sus = 0
            state = DROUGHT if (H[i] <= p.drought_moist and ex < 3 and rate[i] < p.max_rate * .1) else NORMAL
        states[i], scores[i] = state, s
    return rate, drop, states, scores

def hav(la1, lo1, la2, lo2):
    r = 6371
    a = math.sin(math.radians(la2 - la1) / 2) ** 2 + math.cos(math.radians(la1)) * \
        math.cos(math.radians(la2)) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))

clock = lambda m: f"{14 + m // 60:02d}:{m % 60:02d}"

# ===================== 4) ARAYÜZ =====================
st.title("🌲 AnkaGrid – Karar Destek Paneli")
st.markdown("**Ormanın Isı Nabzını Dinleyen Erken Uyarı Ağı**")
st.caption("Orman Şefliği ekranı • ⚠️ Simülasyon: veriler sentetiktir, eşikler pilot alanda kalibre edilecektir.")

with st.sidebar:
    st.header("⚙️ Simülasyon")
    tree = st.selectbox("Ağaç türü (TreeProfile)", list(PROFILES))
    scn = st.radio("Senaryo", ["Normal gün", "Kuraklık stresi", "Kuru ağaçta yangın başlangıcı"], index=2)
    base = PROFILES[tree]
    with st.expander("Eşikleri ayarla (X, Y, Z)"):
        x = st.slider("X: Kritik gövde ısısı (°C)", 40.0, 80.0, float(base.critical_temp))
        y = st.slider("Y: Kritik iç nem (%)", 10.0, 50.0, float(base.critical_moist))
        z = st.slider("Z: ΔT/Δt eşiği (°C/dk)", 0.5, 6.0, float(base.max_rate))
    wind_spd = st.slider("Rüzgâr hızı (km/s)", 0, 60, 25)
    wind_dir = st.slider("Rüzgârın geldiği yön (°)", 0, 359, 270)
    minute = st.slider("⏱ Zaman (dk)", 0, 119, 90)

prof = replace(base, critical_temp=x, critical_moist=y, max_rate=z)
df = simulate(scn)
rate, drop, states, scores = evaluate(df, prof)
i = minute
state = int(states[i])
name, color = STATES[state]

banner = {NORMAL: st.success, DROUGHT: st.warning, WATCH: st.warning, FIRE: st.error}[state]
banner(f"**Düğüm EH-0147 – {name}** | Tür: {tree} ({prof.species})")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Gövde ısısı", f"{df['Gövde °C'][i]:.1f} °C", f"Eşik {prof.critical_temp:.0f}", delta_color="off")
c2.metric("İç nem", f"%{df['İç nem %'][i]:.1f}", f"Kritik %{prof.critical_moist:.0f}", delta_color="off")
c3.metric("ΔT/Δt", f"{rate[i]:+.2f} °C/dk", f"Eşik {prof.max_rate:.1f}", delta_color="off")
c4.metric("Güven skoru", f"%{scores[i] * 20}", f"{scores[i]}/5 sinyal", delta_color="off")

alert_idx = np.where(states == FIRE)[0]
watch_idx = np.where(np.isin(states, [WATCH, FIRE]))[0]
if len(alert_idx) and i >= alert_idx[0]:
    st.info(f"⏱ İlk sinyal: **{clock(int(watch_idx[0]))}** → Alarm: **{clock(int(alert_idx[0]))}** | "
            f"Evre: **KULUÇKA** (gövde {df['Gövde °C'][alert_idx[0]]:.0f} °C, alev/duman henüz yok)")

left, right = st.columns([3, 2])
with left:
    st.subheader("🗺️ CBS Harita")
    rng = np.random.default_rng(3)
    nodes = pd.DataFrame({"lat": np.r_[NODE_LAT, NODE_LAT + rng.uniform(-.02, .02, 11)],
                          "lon": np.r_[NODE_LON, NODE_LON + rng.uniform(-.02, .02, 11)],
                          "color": [color] + ["#f1c40f"] * 2 + ["#2ecc71"] * 9,
                          "size": [450] + [160] * 11})
    st.map(nodes, latitude="lat", longitude="lon", color="color", size="size")
    st.caption("🟢 Normal  🟡 Kuraklık  🟠 İzleme  🔴 Yangın")
with right:
    st.subheader("🌬️ Rüzgâr ve Yayılım")
    dirs = ["K", "KD", "D", "GD", "G", "GB", "B", "KB"]
    frm, to = dirs[round(wind_dir / 45) % 8], dirs[(round(wind_dir / 45) + 4) % 8]
    st.write(f"{wind_spd} km/s, **{frm}** yönünden esiyor → yayılım yönü: **{to}**")
    st.subheader("🚒 Rota ve Kaynak")
    pts = [("Kumluca Müdahale Ekibi", -.04, -.05), ("Finike Ekibi", -.07, .08),
           ("Su kaynağı (gölet)", .008, -.006), ("Helikopter pisti", .02, .02)]
    rows = []
    for nm, dla, dlo in pts:
        d = hav(NODE_LAT, NODE_LON, NODE_LAT + dla, NODE_LON + dlo)
        rows.append({"Hedef": nm, "Mesafe (km)": round(d * 1.35, 1), "Süre (dk)": round(d * 1.35 / 40 * 60)})
    st.table(pd.DataFrame(rows).set_index("Hedef"))

st.subheader("📈 Trend (son 2 saat)")
a, b = st.columns(2)
a.line_chart(df.iloc[: i + 1].set_index("dk")[["Gövde °C", "Ortam °C"]])
b.line_chart(df.iloc[: i + 1].set_index("dk")[["İç nem %"]])

st.subheader("Aksiyonlar")
k1, k2, k3, k4 = st.columns(4)
if k1.button("📞 Ekip Yönlendir", use_container_width=True): st.toast("Ekip yönlendirildi (simülasyon)")
if k2.button("📡 112/AFAD Bildir", use_container_width=True): st.toast("112/AFAD bilgilendirildi (simülasyon)")
if k3.button("✔ Yanlış Alarm", use_container_width=True): st.toast("Geri bildirim modele kaydedildi (simülasyon)")
if k4.button("📄 Rapor", use_container_width=True): st.toast("Olay raporu oluşturuldu (simülasyon)")

st.caption("Sistem sağlığı: Pil %87 • Mesh bağlantısı: 12/12 düğüm • Son nabız: 2 dk önce (simüle)")
