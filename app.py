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
    .stApp {
        background: linear-gradient(135deg, rgba(244, 248, 255, 0.9), rgba(255, 247, 252, 0.9));
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, rgba(27, 72, 128, 0.12), rgba(255, 255, 255, 0.85));
    }
    [data-testid="stSidebar"] .stNumberInput input {
        font-size: 1.05rem;
        font-weight: 600;
    }
    [data-testid="stSidebar"] .value-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 0.35rem;
    }
    [data-testid="stSidebar"] .value-chip {
        background: rgba(27, 72, 128, 0.12);
        color: #1a3c6b;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .stat-list {
        list-style: none;
        padding-left: 0;
        margin: 0;
    }
    .stat-list li {
        font-size: 1.15rem;
        font-weight: 500;
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0.65rem;
        margin-bottom: 0.4rem;
        border-radius: 0.6rem;
        background: rgba(255, 255, 255, 0.65);
        box-shadow: 0 4px 18px rgba(26, 60, 107, 0.08);
    }
    .stat-list li span.label {
        color: #15355e;
        letter-spacing: 0.01em;
    }
    .stat-list li span.value {
        color: #0d2140;
        font-family: "Fira Code", "Source Code Pro", monospace;
    }
    .stPlotlyChart, .stVegaLiteChart, .stPyplot {
        border-radius: 1rem;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.75);
        box-shadow: 0 12px 40px rgba(21, 53, 94, 0.12);
    }
    .stRadio > label {
        font-weight: 600;
    }
    .metric-subheader {
        font-size: 1.05rem;
        letter-spacing: 0.03em;
        color: #173a66;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DB_PATH = Path("data.db")
DB_LOCK = threading.Lock()

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entries(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            value REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    return conn

def add_value(v: float):
    with DB_LOCK:
        conn = get_conn()
        conn.execute("INSERT INTO entries(value, created_at) VALUES(?, ?)", (float(v), datetime.utcnow().isoformat()))
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
            df = pd.read_sql_query("SELECT * FROM entries ORDER BY id DESC LIMIT ?", conn, params=(limit,))
            df = df.iloc[::-1].reset_index(drop=True)
        else:
            df = pd.read_sql_query("SELECT * FROM entries ORDER BY id ASC", conn)
        conn.close()
    return df

# ------------- Sidebar (sterowanie) -------------
def format_value(value: float, decimals: int = 3) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and (np.isnan(value) or not np.isfinite(value))):
        return "—"
    formatted = f"{float(value):.{decimals}f}"
    formatted = formatted.rstrip("0").rstrip(".")
    return formatted if formatted else "0"


with st.sidebar:
    st.header("⚙️ Ustawienia")
    var_label = st.text_input("Etykieta zmiennej", value="Godziny snu")

    number = st.number_input("Twoja wartość", value=None, placeholder="np. 7.5", step=0.1, format="%.3f")
    col_add1, col_add2 = st.columns([1,1])
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
    st.caption("Auto-odświeżanie powoduje automatyczne pobieranie nowych wpisów (wszyscy widzą to samo).")
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
        chips = "".join(
            f"<span class='value-chip'>{format_value(v, 2)}</span>"
            for v in reversed(recent_values)
        )
        st.markdown(f"<div class='value-chips'>{chips}</div>", unsafe_allow_html=True)

# ------------- Nagłówek -------------
st.title("📊 Żywy histogram — wspólne zbieranie danych")
st.markdown(f"**Zmienna:** _{var_label}_")

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
                    # Skaluje gęstość do skali histogramu (wysokości słupków)
                    counts, edges = np.histogram(x, bins=bins)
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
        # kurtoza nadwyżkowa (Fisher=True): 0 dla rozkł. normalnego
        stats["Kurtosis (excess)"] = kurtosis(x, fisher=True, bias=False) if x.size > 3 else np.nan
        stats["Min"] = np.nanmin(x)
        stats["Max"] = np.nanmax(x)

        stat_items = []
        for key, value in stats.items():
            if isinstance(value, (int, np.integer)):
                display_value = f"{int(value)}"
            else:
                display_value = format_value(value, 3)
            stat_items.append((key, display_value))

        st.markdown(
            "<ul class='stat-list'>" +
            "".join(
                f"<li><span class='label'>{label}</span><span class='value'>{val}</span></li>"
                for label, val in stat_items
            ) +
            "</ul>",
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
