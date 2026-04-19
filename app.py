"""
app.py — ETF Portfolio Dashboard
Interfaccia Streamlit per analizzare portafogli di ETF.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import download_prices
from src.portfolio import (
    portfolio_cumulative_value,
    portfolio_summary,
    correlation_matrix,
)

# ==========================================================================
# CONFIGURAZIONE PAGINA
# ==========================================================================
st.set_page_config(
    page_title="ETF Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 ETF Portfolio Dashboard")
st.caption(
    "Analizza la performance storica di un portafoglio di ETF: "
    "rendimenti, volatilità, Sharpe ratio e drawdown."
)

# ==========================================================================
# SIDEBAR — INPUT UTENTE
# ==========================================================================
st.sidebar.header("⚙️ Portfolio Setup")

# Input ticker (separati da virgola)
tickers_input = st.sidebar.text_input(
    "Ticker (separati da virgola)",
    value="VWCE.DE, AGGH.MI",
    help="Es: VWCE.DE, AGGH.MI, IWDA.AS. Usa i ticker Yahoo Finance.",
)
tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

# Input pesi
weights_input = st.sidebar.text_input(
    "Pesi (somma = 1)",
    value="0.6, 0.4",
    help="Es: 0.6, 0.4 per un 60/40.",
)
try:
    weights = [float(w.strip()) for w in weights_input.split(",") if w.strip()]
except ValueError:
    st.sidebar.error("Pesi non validi. Inserisci numeri separati da virgole.")
    st.stop()

# Date
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input(
    "Inizio",
    value=pd.Timestamp("2020-01-01"),
    min_value=pd.Timestamp("2000-01-01"),
)
end_date = col2.date_input(
    "Fine",
    value=pd.Timestamp.today(),
)

# Risk-free
risk_free = st.sidebar.slider(
    "Risk-free rate (%)",
    min_value=0.0,
    max_value=6.0,
    value=3.0,
    step=0.25,
    help="Tasso risk-free annualizzato usato per lo Sharpe ratio.",
) / 100

# Strategia ribilanciamento
rebalance = st.sidebar.radio(
    "Strategia",
    options=["Rebalanced (constant-mix)", "Buy & Hold"],
    index=0,
)
is_rebalanced = rebalance.startswith("Rebalanced")

# Bottone di avvio
run_button = st.sidebar.button("🚀 Analizza Portfolio", type="primary")

# ==========================================================================
# ELABORAZIONE
# ==========================================================================
if run_button:
    # Validazione pesi
    if len(tickers) != len(weights):
        st.error(
            f"❌ Numero ticker ({len(tickers)}) diverso dal numero "
            f"di pesi ({len(weights)})."
        )
        st.stop()

    if not (0.999 < sum(weights) < 1.001):
        st.error(f"❌ I pesi sommano a {sum(weights):.3f}, devono sommare a 1.")
        st.stop()

    # Scarica dati
    with st.spinner(f"📥 Scarico dati per {', '.join(tickers)}..."):
        try:
            prices = download_prices(
                tickers,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
            )
        except ValueError as e:
            st.error(f"❌ Errore nel download: {e}")
            st.stop()

    st.success(f"✅ Scaricati {len(prices)} giorni di trading.")

    # ============================================================
    # METRICHE PRINCIPALI
    # ============================================================
    st.header("📊 Metriche del portafoglio")

    summary = portfolio_summary(
        prices, weights, risk_free_rate=risk_free, rebalance=is_rebalanced
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CAGR", f"{summary['CAGR'] * 100:.2f}%")
    col2.metric("Volatility", f"{summary['Volatility'] * 100:.2f}%")
    col3.metric("Sharpe Ratio", f"{summary['Sharpe Ratio']:.2f}")
    col4.metric("Max Drawdown", f"{summary['Max Drawdown'] * 100:.2f}%")

    # ============================================================
    # GRAFICO PERFORMANCE
    # ============================================================
    st.header("📈 Performance storica")

    value = portfolio_cumulative_value(prices, weights, initial_value=100)
    value_df = value.reset_index()
    value_df.columns = ["Date", "Valore Portafoglio (€)"]

    fig = px.line(
        value_df,
        x="Date",
        y="Valore Portafoglio (€)",
        title=f"Valore del portafoglio (partendo da €100)",
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # PREZZI SINGOLI ETF
    # ============================================================
    st.header("📉 Prezzi dei singoli ETF (normalizzati a 100)")

    # Normalizziamo ogni ETF a 100 al giorno iniziale per confronto visivo
    prices_normalized = prices / prices.iloc[0] * 100
    st.line_chart(prices_normalized, use_container_width=True)

    # ============================================================
    # MATRICE CORRELAZIONE
    # ============================================================
    if len(tickers) > 1:
        st.header("🔗 Matrice di correlazione")
        corr = correlation_matrix(prices).round(3)

        fig_corr = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # ============================================================
    # TABELLA PREZZI (EXPANDER)
    # ============================================================
    with st.expander("📋 Mostra tabella prezzi"):
        st.dataframe(prices.tail(50), use_container_width=True)

else:
    st.info("👈 Imposta il tuo portafoglio nella sidebar e clicca **Analizza Portfolio**.")