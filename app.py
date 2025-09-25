import sqlite3
import threading
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, gaussian_kde
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# ------------- Konfiguracja podstawowa -------------
st.set_page_config(page_title="Żywy histogram (Streamlit)", page_icon="📊", layout="wide")

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
with st.sidebar:
    st.header("⚙️ Ustawienia")
    var_label = st.text_input("Etykieta zmiennej", value="Godziny snu")

    number = st.number_input("Twoja wartość", value=None, placeholder="np. 7.5", step=0.1, format="%.6f")
    col_add1, col_add2 = st.columns([1,1])
    with col_add1:
        add_btn = st.button("➕ Dodaj wartość", use_container_width=True)
    with col_add2:
        refresh_btn = st.button("🔄 Odśwież teraz", use_container_width=True)

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
    st.experimental_rerun()

if auto_refresh:
    st_autorefresh(interval=2000, limit=None, key="auto_refresh_key")

# ------------- Dane -------------
df = read_values()
x = df["value"].to_numpy(dtype=float) if not df.empty else np.array([])

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

        out = pd.DataFrame({k: [np.round(v, 4) if isinstance(v, (int, float, np.floating)) else v]
                            for k, v in stats.items()})
        st.table(out)

    st.subheader("Ostatnie wpisy")
    df_last = read_values(limit=10)
    if df_last.empty:
        st.write("—")
    else:
        st.write(", ".join(f"{v:.4g}" for v in df_last["value"]))

st.divider()
with st.expander("ℹ️ Informacje techniczne / prywatność"):
    st.markdown(
        "- Dane to wyłącznie **pojedyncze liczby** podawane przez uczestników.\n"
        "- Brak zbierania danych osobowych.\n"
        "- Baza: **SQLite** w pliku `data.db` w katalogu aplikacji (czyści się po resecie).\n"
        "- Na bezpłatnym hostingu Streamlit plik jest trwały **tylko podczas życia instancji**; po restarcie hosta stan może zostać wyzerowany.\n"
        "- Do zajęć na żywo to w zupełności wystarcza; do trwałego zapisu rozważ zewnętrzną bazę (np. PostgreSQL/Supabase)."
    )
