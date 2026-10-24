"""
===============================================================================
TÜRKÇE OSU (MLU) AKADEMİK KLİNİK ARAŞTIRMA PANELİ (Streamlit v3.0)
===============================================================================
Referans: Ege, P., Acarlar, F., & Güleryüz, F. (1998).
          Türk Çocuklarında Ortalama Sözce Uzunluğunun (OSU) Gelişimi ve Yaşla İlişkisi.
          Türk Psikoloji Dergisi, 13(41), 19-31.
===============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# =============================================================================
# 1. SAYFA YAPILANDIRMASI
# =============================================================================
st.set_page_config(
    page_title="Türkçe OSU (MLU) Akademik Analiz Paneli",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 2. ÇEKİRDEK KÜTÜPHANE MİMARİSİ (v3.0)
# =============================================================================

MIN_AGE_MONTHS = 17.0
MAX_AGE_MONTHS = 59.0
MIN_OBSERVED_OSU = 0.0
MAX_OBSERVED_OSU = 15.0
MIN_AGE_MODEL_B_OSU = 1.0
MAX_AGE_MODEL_B_OSU = 7.0

MODEL_A_SLOPE = 0.110
MODEL_A_INTERCEPT = -0.008
MODEL_A_SD_SLOPE = 0.008
MODEL_A_SD_INTERCEPT = 0.551

MODEL_B_SLOPE = 6.351
MODEL_B_INTERCEPT = 12.236

OSU_AGE_SD_TABLE: tuple[tuple[float, float, float], ...] = (
    (1.0, 18.587, 2.0),
    (1.5, 21.762, 2.8),
    (2.0, 24.938, 4.0),
    (3.0, 31.289, 6.0),
    (4.0, 37.640, 7.5),
    (5.0, 43.991, 8.5),
    (6.0, 50.342, 8.5),
    (7.0, 56.693, 3.0),
)

@dataclass(frozen=True)
class Prediction:
    value: float
    lower: float
    upper: float
    sd: float
    z_score: Optional[float] = None
    percentile: Optional[float] = None
    is_extrapolated: bool = False

def _norm_cdf(z: float) -> float:
    result = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return min(1.0, max(0.0, result))

def _interpolate_sd(osu: float) -> float:
    table = OSU_AGE_SD_TABLE
    if osu <= table[0][0]:
        return float(table[0][2])
    if osu >= table[-1][0]:
        return float(table[-1][2])
    for left, right in zip(table, table[1:]):
        x1, _, sd1 = left
        x2, _, sd2 = right
        if x1 <= osu <= x2:
            ratio = (osu - x1) / (x2 - x1)
            return float(sd1 + ratio * (sd2 - sd1))
    return float(table[-1][2])

def osu_from_age(age_months: float, observed_osu: Optional[float] = None, allow_extrapolation: bool = False) -> Prediction:
    outside = not (MIN_AGE_MONTHS <= age_months <= MAX_AGE_MONTHS)
    if outside and not allow_extrapolation:
        raise ValueError("Yaş 17–59 ay aralığında olmalıdır.")
    mean = MODEL_A_SLOPE * age_months + MODEL_A_INTERCEPT
    sd = MODEL_A_SD_SLOPE * age_months + MODEL_A_SD_INTERCEPT
    z_score, percentile = None, None
    if observed_osu is not None:
        z_score = (float(observed_osu) - mean) / sd
        percentile = round(_norm_cdf(z_score) * 100.0, 1)
    return Prediction(
        value=round(mean, 2),
        lower=round(mean - sd, 2),
        upper=round(mean + sd, 2),
        sd=round(sd, 3),
        z_score=round(z_score, 2) if z_score is not None else None,
        percentile=percentile,
        is_extrapolated=outside
    )

def age_from_osu(osu: float, observed_age: Optional[float] = None) -> Prediction:
    mean = MODEL_B_SLOPE * osu + MODEL_B_INTERCEPT
    sd = _interpolate_sd(osu)
    z_score, percentile = None, None
    if observed_age is not None:
        z_score = (observed_age - mean) / sd
        percentile = round(_norm_cdf(z_score) * 100.0, 1)
    return Prediction(
        value=round(mean, 1),
        lower=round(mean - sd, 1),
        upper=round(mean + sd, 1),
        sd=round(sd, 1),
        z_score=round(z_score, 2) if z_score is not None else None,
        percentile=percentile,
        is_extrapolated=False
    )

# =============================================================================
# 3. STREAMLIT ARAYÜZ TASARIMI
# =============================================================================

st.title("📊 Türkçe Ortalama Sözce Uzunluğu (OSU / MLU) Araştırma Paneli")
st.markdown("""
Bu uygulama, **Ege, Acarlar & Güleryüz (1998)** normatif regresyon modellerini temel alarak 
klinik değerlendirmeler ve dilbilimsel araştırmalar için hassas istatistiksel kestirimler sunar.
""")

# Kenar Çubuğu Navigasyonu
st.sidebar.header("🛠️ Analiz Modu")
analysis_mode = st.sidebar.radio(
    "Çalışma Alanı Seçin:",
    ["Model A: Yaşten OSU'ya (İleri Yönlü)", "Model B: OSU'dan Yaşa (Geri Yönlü)", "Toplu Veri (Batch) Analizi", "Normatif Eğri Görselleştirici"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Metodolojik Not:** Model A ve Model B birbirinin matematiksel tersi değildir. Hangi yönü kullandığınızı raporlarınızda belirtiniz.")

# -----------------------------------------------------------------------------
# MOD 1: MODEL A (Yaş -> OSU)
# -----------------------------------------------------------------------------
if analysis_mode == "Model A: Yaşten OSU'ya (İleri Yönlü)":
    st.header("Model A: Kronolojik Yaştan Beklenen OSU Kestirimi")
    
    col1, col2 = st.columns(2)
    with col1:
        age_input = st.number_input("Çocuğun Yaşı (Ay cinsinden)", min_value=12.0, max_value=72.0, value=36.0, step=1.0)
        allow_extrap = st.checkbox("17–59 ay norm sınırı dışı hesaplamaya (extrapolation) izin ver", value=False)
    with col2:
        has_observed = st.checkbox("Gözlenen (Ölçülen) OSU Değerini Gir", value=True)
        observed_osu_input = st.number_input("Gözlenen OSU", min_value=0.1, max_value=15.0, value=3.2, step=0.1) if has_observed else None

    if st.button("Model A Hesapla ve Raporla", type="primary"):
        try:
            pred = osu_from_age(age_months=age_input, observed_osu=observed_osu_input, allow_extrapolation=allow_extrap)
            
            st.success("Hesaplama başarıyla tamamlandı.")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Beklenen OSU (Ortalama)", f"{pred.value:.2f}")
            m2.metric("±1 SS Alt Sınır", f"{pred.lower:.2f}")
            m3.metric("±1 SS Üst Sınır", f"{pred.upper:.2f}")
            m4.metric("Hesaplanan SS", f"{pred.sd:.3f}")
            
            if pred.z_score is not None:
                st.markdown("### 📈 Bireysel Performans ve Normatif Karşılaştırma")
                c1, c2 = st.columns(2)
                c1.metric("Z-Skoru ($Z$)", f"{pred.z_score:+.2f}")
                c2.metric("Persentil Dilimi", f"%{pred.percentile}")
                
                # Klinik Durum Eşiği
                if pred.z_score < -1.5:
                    st.warning("⚠️ **Klinik Durum:** Gözlenen OSU beklentinin belirgin altındadır (Z < -1.5). Ayrıntılı DKT değerlendirmesi önerilir.")
                elif pred.z_score > 1.5:
                    st.info("ℹ️ **Klinik Durum:** Gözlenen OSU beklentinin belirgin üzerindedir (Z > +1.5). İleri düzey dil geliştirme göstergesi olabilir.")
                else:
                    st.info("✅ **Klinik Durum:** Gözlenen OSU normal ranj içindedir (±1.0 SS aralığı).")

            if pred.is_extrapolated:
                st.warning("⚠️ **Uyarı:** Girilen yaş değeri özgün örneklem aralığı olan 17–59 ay dışındadır.")

        except ValueError as e:
            st.error(f"Geçersiz Girdi Hatası: {e}")

# -----------------------------------------------------------------------------
# MOD 2: MODEL B (OSU -> Yaş)
# -----------------------------------------------------------------------------
elif analysis_mode == "Model B: OSU'dan Yaşa (Geri Yönlü)":
    st.header("Model B: Gözlenen OSU Değerinden Gelişimsel Yaş Kestirimi")
    
    col1, col2 = st.columns(2)
    with col1:
        osu_input = st.number_input("Gözlenen OSU Değeri", min_value=1.0, max_value=7.0, value=3.0, step=0.1)
    with col2:
        has_obs_age = st.checkbox("Çocuğun Gerçek Yaşını Gir (Z-skoru için)", value=True)
        obs_age_input = st.number_input("Gerçek Kronolojik Yaş (Ay)", min_value=17.0, max_value=59.0, value=30.0, step=1.0) if has_obs_age else None

    if st.button("Model B Hesapla", type="primary"):
        try:
            pred = age_from_osu(osu=osu_input, observed_age=obs_age_input)
            st.success("Hesaplama tamamlandı.")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Tahmini Gelişim Yaşı", f"{pred.value:.1f} ay")
            m2.metric("±1 SD Alt Sınır", f"{pred.lower:.1f} ay")
            m3.metric("±1 SD Üst Sınır", f"{pred.upper:.1f} ay")
            
            if pred.z_score is not None:
                st.markdown("### 📈 Yaş Ölçeğinde Sapma Analizi")
                c1, c2 = st.columns(2)
                c1.metric("Z-Skoru", f"{pred.z_score:+.2f}")
                c2.metric("Persentil", f"%{pred.percentile}")
        except ValueError as e:
            st.error(f"Hata: {e}")

# -----------------------------------------------------------------------------
# MOD 3: TOPLU VERİ (BATCH) ANALİZİ
# -----------------------------------------------------------------------------
elif analysis_mode == "Toplu Veri (Batch) Analizi":
    st.header("📁 Araştırma Veri Setleri İçin Toplu İşleme")
    st.markdown("Birden fazla katılımcının verilerini içeren bir CSV dosyası yükleyerek toplu analiz yapabilirsiniz.")
    
    # Örnek şablon indirme
    sample_data = pd.DataFrame({
        "Katilimci_ID": ["K01", "K02", "K03"],
        "Yas_Ay": [24, 36, 48],
        "Gozlenen_OSU": [2.1, 3.8, 5.5]
    })
    csv_bytes = sample_data.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Örnek CSV Şablonunu İndir", data=csv_bytes, file_name="osu_ornek_veri.csv", mime="text/csv")
    
    uploaded_file = st.file_uploader("CSV Veri Dosyanızı Yükleyin", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Yüklenen Veri Önizlemesi:", df.head())
        
        if st.button("Toplu Analizi Çalıştır", type="primary"):
            results = []
            for _, row in df.iterrows():
                kid = row.get("Katilimci_ID", "Bilinmeyen")
                y = row.get("Yas_Ay")
                o = row.get("Gozlenen_OSU")
                try:
                    p = osu_from_age(y, observed_osu=o, allow_extrapolation=True)
                    results.append({
                        "Katilimci_ID": kid,
                        "Yas_Ay": y,
                        "Gozlenen_OSU": o,
                        "Beklenen_OSU": p.value,
                        "SS": p.sd,
                        "Z_Skoru": p.z_score,
                        "Persentil": p.percentile
                    })
                except Exception as e:
                    results.append({"Katilimci_ID": kid, "Hata": str(e)})
            
            res_df = pd.DataFrame(results)
            st.dataframe(res_df)
            
            res_csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Analiz Sonuçlarını İndir (CSV)", data=res_csv, file_name="osu_analiz_sonuclari.csv", mime="text/csv")

# -----------------------------------------------------------------------------
# MOD 4: NORMATİF EĞRİ GÖRSELLEŞTİRİCİ
# -----------------------------------------------------------------------------
elif analysis_mode == "Normatif Eğri Görselleştirici":
    st.header("📈 Ege et al. (1998) Model A Normatif Regresyon Eğrileri")
    st.markdown("17 ile 59 ay arasındaki yaş aralığı için beklenen ortalama OSU değerini ve $\\pm 1$ SS güven bantlarını interaktif olarak inceleyin.")
    
    # Grafik verisi oluşturma
    ages = list(range(17, 60))
    means, lowers, uppers = [], [], []
    for a in ages:
        p = osu_from_age(float(a))
        means.append(p.value)
        lowers.append(p.lower)
        uppers.append(p.upper)
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ages, y=uppers, mode='lines', name='+1 SS Üst Sınır', line=dict(color='rgba(100,150,250,0.2)')))
    fig.add_trace(go.Scatter(x=ages, y=means, mode='lines', name='Beklenen OSU (Ortalama)', line=dict(color='rgb(31, 119, 180)', width=3)))
    fig.add_trace(go.Scatter(x=ages, y=lowers, mode='lines', name='-1 SS Alt Sınır', line=dict(color='rgba(100,150,250,0.2)'), fill='tonexty', fillcolor='rgba(100,150,250,0.1)'))
    
    fig.update_layout(
        title="Yaş - OSU Normatif Dağılım Bandı",
        xaxis_title="Kronolojik Yaş (Ay)",
        yaxis_title="Ortalama Sözce Uzunluğu (OSU)",
        template="plotly_white",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Türkçe OSU Akademik Analiz Paneli v3.0 | DKT & Araştırma Destek Aracı</div>", unsafe_allow_html=True)