import html
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, kurtosis, skew
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ------------- Konfiguracja podstawowa -------------
st.set_page_config(page_title="Żywy histogram (Streamlit)", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --gradient-primary: linear-gradient(135deg, #2f3cc9, #a334d6);
        --surface: rgba(255, 255, 255, 0.82);
        --surface-strong: rgba(255, 255, 255, 0.95);
        --text-strong: #111a3a;
        --text-muted: #4a5785;
    }
    * {
        font-family: "Inter", "Segoe UI", sans-serif !important;
    }
    .stApp {
        background:
            radial-gradient(circle at 18% 20%, rgba(163, 52, 214, 0.18), transparent 52%),
            radial-gradient(circle at 80% 15%, rgba(47, 60, 201, 0.22), transparent 46%),
            radial-gradient(circle at 20% 85%, rgba(16, 197, 222, 0.12), transparent 45%),
            #f6f7ff;
        color: var(--text-strong);
    }
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background: rgba(14, 22, 62, 0.78);
        backdrop-filter: blur(18px);
        color: #f6f7ff;
    }
    [data-testid="stSidebar"] .stNumberInput input,
    [data-testid="stSidebar"] .stTextInput input {
        border-radius: 0.75rem;
        border: 1px solid rgba(255, 255, 255, 0.32);
        background: rgba(255, 255, 255, 0.16);
        color: inherit;
        font-weight: 600;
    }
    [data-testid="stSidebar"] .stButton button,
    [data-testid="stSidebar"] .stDownloadButton button {
        border-radius: 999px;
        border: none;
        background: var(--gradient-primary);
        color: white;
        font-weight: 700;
        letter-spacing: 0.02em;
        box-shadow: 0 16px 34px rgba(44, 58, 189, 0.35);
    }
    .metric-subheader {
        font-size: 1.05rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgba(255, 255, 255, 0.85);
        margin-bottom: 0.3rem;
    }
    .recent-values {
        display: grid;
        gap: 0.35rem;
        margin-top: 0.5rem;
    }
    .recent-values span {
        background: rgba(255, 255, 255, 0.16);
        color: inherit;
        border-radius: 0.65rem;
        padding: 0.4rem 0.75rem;
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: 0.01em;
    }
    .main-title {
        font-size: clamp(2.4rem, 3.5vw, 3.4rem);
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.4rem;
        color: var(--text-strong);
    }
    .main-title span {
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    .main-subtitle {
        font-size: 1.05rem;
        color: var(--text-muted);
        margin-bottom: 1.5rem;
    }
    .stat-card {
        background: var(--surface);
        border-radius: 1rem;
        padding: 1.35rem 1.45rem;
        box-shadow: 0 20px 45px rgba(20, 26, 60, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(14px);
    }
    .stat-list {
        list-style: none;
        padding-left: 0;
        margin: 0;
        display: grid;
        gap: 0.75rem;
    }
    .stat-list li {
        display: grid;
        gap: 0.35rem;
        align-items: start;
        padding: 1rem 1.1rem;
        border-radius: 0.9rem;
        background: var(--surface-strong);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.85),
                    0 18px 36px rgba(24, 32, 74, 0.16);
    }
    .stat-list li span.label {
        color: var(--text-muted);
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-size: 0.92rem;
    }
    .stat-list li span.value {
        color: var(--text-strong);
        font-family: "Fira Code", "Source Code Pro", monospace;
        font-size: clamp(1.45rem, 3vw, 1.85rem);
        font-weight: 700;
    }
    .stPlotlyChart, .stVegaLiteChart, .stPyplot {
        border-radius: 1.15rem !important;
        padding: 1.25rem !important;
        background: var(--surface);
        box-shadow: 0 28px 60px rgba(20, 26, 60, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.42);
        backdrop-filter: blur(18px);
    }
    .stRadio > label {
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .stDivider {
        margin: 1.4rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DB_PATH = Path("data.db")
DB_LOCK = threading.Lock()


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS entries(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            value REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    return conn


def add_value(v: float):
    with DB_LOCK:
        conn = get_conn()
        conn.execute(
            "INSERT INTO entries(value, created_at) VALUES(?, ?)",
            (float(v), datetime.utcnow().isoformat()),
        )
        conn.commit()
        conn.close()


def clear_values():
    with DB_LOCK:
        conn = get_conn()
        conn.execute("DELETE FROM entries")
        conn.commit()
        conn.close()


def read_values(limit: int | None = None) -> pd.DataFrame:
    with DB_LOCK:
        conn = get_conn()
        if limit:
            df = pd.read_sql_query(
                "SELECT * FROM entries ORDER BY id DESC LIMIT ?",
                conn,
                params=(limit,),
            )
            df = df.iloc[::-1].reset_index(drop=True)
        else:
            df = pd.read_sql_query("SELECT * FROM entries ORDER BY id ASC", conn)
        conn.close()
    return df


# ------------- Sidebar (sterowanie) -------------
def format_value(value: float | int | None, decimals: int = 3) -> str:
    if value is None:
        return "—"

    if isinstance(value, (float, np.floating)) and (np.isnan(value) or not np.isfinite(value)):
        return "—"

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "—"

    if decimals == 0:
        return f"{int(round(numeric_value))}"

    formatted = f"{numeric_value:.{decimals}f}"
    formatted = formatted.rstrip("0").rstrip(".")
    return formatted if formatted else "0"


with st.sidebar:
    st.header("⚙️ Ustawienia")
    var_label = st.text_input("Etykieta zmiennej", value="Godziny snu")

    number = st.number_input(
        "Twoja wartość",
        value=None,
        placeholder="np. 7.5",
        step=0.1,
        format="%.3f",
    )
    col_add1, col_add2 = st.columns([1, 1])
    with col_add1:
        add_btn = st.button("➕ Dodaj wartość", use_container_width=True)
    with col_add2:
        refresh_btn = st.button("🔄 Odśwież teraz", use_container_width=True)

    recent_box = st.container()

    st.divider()
    plot_type = st.radio("Rodzaj wykresu", options=["Histogram", "Boxplot"], horizontal=True)
    if plot_type == "Histogram":
        bins = st.slider("Liczba koszyków", min_value=5, max_value=60, value=15, step=1)
        show_density = st.checkbox("Pokaż krzywą gęstości (KDE)", value=False)
    else:
        bins = None
        show_density = False

    st.divider()
    st.caption(
        "Auto-odświeżanie powoduje automatyczne pobieranie nowych wpisów (wszyscy widzą to samo)."
    )
    auto_refresh = st.toggle("Auto-odświeżanie co 2 sekundy", value=True)

    st.divider()
    st.subheader("🧹 Reset ćwiczenia")
    with st.popover("Resetuj wszystkie dane (z potwierdzeniem)"):
        st.write("Ta operacja usunie **wszystkie** wpisy (nieodwracalna).")
        confirm = st.checkbox("Tak, na pewno chcę usunąć wszystkie wpisy.")
        reset_btn = st.button("❌ Wyczyść teraz", disabled=not confirm, type="secondary")

# ------------- Akcje: Dodanie/Reset/Refresh -------------
if add_btn:
    if number is None:
        st.toast("Wpisz najpierw liczbę.", icon="⚠️")
    else:
        try:
            x = float(number)
            if np.isfinite(x):
                add_value(x)
                st.toast("Dodano wartość ✅", icon="✅")
            else:
                st.toast("Wartość musi być skończoną liczbą.", icon="⚠️")
        except Exception:
            st.toast("Nie udało się dodać wartości.", icon="❌")

if reset_btn:
    clear_values()
    st.toast("Dane wyczyszczone.", icon="🧹")

if refresh_btn:
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

if auto_refresh:
    st_autorefresh(interval=2000, limit=None, key="auto_refresh_key")

# ------------- Dane -------------
df = read_values()
x = df["value"].to_numpy(dtype=float) if not df.empty else np.array([])

with recent_box:
    st.markdown("<p class='metric-subheader'>🗒️ Ostatnie wartości</p>", unsafe_allow_html=True)
    if df.empty:
        st.write("—")
    else:
        recent_values = df.tail(10)["value"].tolist()
        items = "".join(
            f"<span>{format_value(v, 2)}</span>"
            for v in reversed(recent_values)
        )
        st.markdown(f"<div class='recent-values'>{items}</div>", unsafe_allow_html=True)

# ------------- Nagłówek -------------
var_label_display = html.escape(var_label)
st.markdown(
    f"<h1 class='main-title'>Rozkład zmiennej: <span>{var_label_display}</span></h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='main-subtitle'>Podglądaj bieżące dane uczestników na żywo w estetycznym ujęciu.</p>",
    unsafe_allow_html=True,
)

# ------------- Główna siatka -------------
left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("Wizualizacja")
    if x.size == 0:
        st.info("Brak danych. Poproś studentów o wpisanie pierwszych wartości po lewej stronie.")
    else:
        fig = plt.figure(figsize=(8, 4.8))
        ax = plt.gca()

        if plot_type == "Histogram":
            ax.hist(x, bins=bins, edgecolor="black")
            ax.set_title(f"Histogram — {var_label}")
            ax.set_xlabel(var_label)
            ax.set_ylabel("Liczebność")

            if show_density and x.size >= 2 and np.all(np.isfinite(x)):
                try:
                    kde = gaussian_kde(x)
                    xs = np.linspace(x.min(), x.max(), 500)
                    ys = kde(xs)
                    counts, _ = np.histogram(x, bins=bins)
                    scale = counts.max() / ys.max() if ys.max() > 0 else 1.0
                    ax.plot(xs, ys * scale, linewidth=2)
                except Exception:
                    pass

        else:  # Boxplot
            ax.boxplot(x, vert=False, whis=1.5)
            ax.set_title(f"Boxplot — {var_label}")
            ax.set_xlabel(var_label)

        st.pyplot(fig, use_container_width=True)

with right:
    st.subheader("Statystyki opisowe")
    if x.size == 0:
        st.write("—")
    else:
        stats = {}
        stats["N"] = int(x.size)
        stats["Mean"] = np.nanmean(x)
        stats["Median"] = np.nanmedian(x)
        stats["Variance"] = np.nanvar(x, ddof=1) if x.size > 1 else np.nan
        stats["SD"] = np.nanstd(x, ddof=1) if x.size > 1 else np.nan
        stats["Skewness"] = skew(x, bias=False) if x.size > 2 else np.nan
        stats["Kurtosis (excess)"] = kurtosis(x, fisher=True, bias=False) if x.size > 3 else np.nan
        stats["Min"] = np.nanmin(x)
        stats["Max"] = np.nanmax(x)

        stat_precision = {
            "N": 0,
            "Mean": 2,
            "Median": 2,
            "Variance": 3,
            "SD": 2,
            "Skewness": 3,
            "Kurtosis (excess)": 3,
            "Min": 2,
            "Max": 2,
        }

        stat_items = []
        for key, value in stats.items():
            decimals = stat_precision.get(key, 3)
            display_value = format_value(value, decimals)
            stat_items.append((key, display_value))

        st.markdown(
            "<div class='stat-card'><ul class='stat-list'>" +
            "".join(
                f"<li><span class='label'>{label}</span><span class='value'>{val}</span></li>"
                for label, val in stat_items
            ) +
            "</ul></div>",
           unsafe_allow_html=True,
        )

st.divider()
with st.expander("ℹ️ Informacje techniczne / prywatność"):
    st.markdown(
        "- Dane to wyłącznie **pojedyncze liczby** podawane przez uczestników.\n"
        "- Brak zbierania danych osobowych.\n"
        "- Baza: **SQLite** w pliku `data.db` w katalogu aplikacji (czyści się po resecie).\n"
        "- Na bezpłatnym hostingu Streamlit plik jest trwały **tylko podczas życia instancji**; po restarcie hosta stan może zostać wyzerowany.\n"
        "- Do zajęć na żywo to w zupełności wystarcza; do trwałego zapisu rozważ zewnętrzną bazę (np. PostgreSQL/Supabase)."
    )
