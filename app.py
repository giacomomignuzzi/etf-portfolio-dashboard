"""
app.py — ETF Portfolio Dashboard
Interfaccia Streamlit per analizzare portafogli di ETF.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import download_prices
from src.metrics import alpha_beta, daily_returns
from src.portfolio import (
    portfolio_cumulative_value,
    portfolio_summary,
    portfolio_returns,
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

# Benchmark opzionale
st.sidebar.markdown("---")
st.sidebar.subheader("📏 Benchmark (opzionale)")
benchmark_ticker = st.sidebar.text_input(
    "Ticker benchmark",
    value="",
    placeholder="Es: ^GSPC, VWCE.DE, URTH",
    help=(
        "Indice o ETF di riferimento. "
        "Indici: ^GSPC (S&P 500), ^STOXX50E (Euro Stoxx), ^IXIC (Nasdaq). "
        "ETF: VWCE.DE, IWDA.AS, CSPX.MI. "
        "Lascia vuoto per non confrontare."
    ),
).strip()

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

    # Mostra info dettagliate sul periodo coperto
    actual_start = prices.index.min().date()
    actual_end = prices.index.max().date()
    n_days = len(prices)
    n_years = (actual_end - actual_start).days / 365.25

    st.success(
        f"✅ Scaricati **{n_days} giorni** di trading "
        f"dal **{actual_start}** al **{actual_end}** "
        f"(~{n_years:.1f} anni)."
    )

    # Warning se il periodo effettivo è molto più corto di quello richiesto
    requested_start = start_date
    if (actual_start - requested_start).days > 30:
        st.warning(
            f"⚠️ **Attenzione**: hai richiesto dati dal {requested_start}, "
            f"ma almeno uno degli ETF ha storia limitata. "
            f"Il periodo effettivamente analizzato inizia dal **{actual_start}**. "
            f"Prova a escludere gli ETF più giovani o a scegliere una data di inizio più recente."
        )

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
    # CONFRONTO CON BENCHMARK
    # ============================================================
    benchmark_prices = None  # inizializzo per uso successivo

    if benchmark_ticker:
        st.header("📏 Confronto con benchmark")

        try:
            benchmark_prices = download_prices(
                [benchmark_ticker],
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
            )
        except ValueError as e:
            st.warning(f"⚠️ Impossibile scaricare il benchmark: {e}")
            benchmark_prices = None

        if benchmark_prices is not None:
            # Calcoliamo i rendimenti del portafoglio e del benchmark
            port_returns = portfolio_returns(
                prices.dropna(), weights, rebalance=is_rebalanced
            )
            bench_returns = daily_returns(benchmark_prices[benchmark_ticker])

            # Metriche del benchmark (trattato come "portfolio" a pesi [1.0])
            bench_summary = portfolio_summary(
                benchmark_prices, [1.0], risk_free_rate=risk_free, rebalance=False
            )

            # Alpha e beta
            try:
                ab = alpha_beta(port_returns, bench_returns, risk_free_rate=risk_free)

                # Tabella comparativa
                comparison_df = pd.DataFrame({
                    "Portfolio": [
                        f"{summary['CAGR'] * 100:.2f}%",
                        f"{summary['Volatility'] * 100:.2f}%",
                        f"{summary['Sharpe Ratio']:.2f}",
                        f"{summary['Max Drawdown'] * 100:.2f}%",
                    ],
                    f"Benchmark ({benchmark_ticker})": [
                        f"{bench_summary['CAGR'] * 100:.2f}%",
                        f"{bench_summary['Volatility'] * 100:.2f}%",
                        f"{bench_summary['Sharpe Ratio']:.2f}",
                        f"{bench_summary['Max Drawdown'] * 100:.2f}%",
                    ],
                }, index=["CAGR", "Volatility", "Sharpe Ratio", "Max Drawdown"])

                st.dataframe(comparison_df, use_container_width=True)

                # Alpha, Beta, Correlation (CAPM)
                col_a, col_b, col_c = st.columns(3)
                col_a.metric(
                    "Alpha (annualizzato)",
                    f"{ab['alpha'] * 100:.2f}%",
                    help=(
                        "Rendimento in eccesso rispetto a quanto atteso dato il beta. "
                        "Positivo = il portafoglio ha 'battuto' il benchmark risk-adjusted."
                    ),
                )
                col_b.metric(
                    "Beta",
                    f"{ab['beta']:.2f}",
                    help=(
                        "Sensibilità al benchmark. Beta=1 si muove come il mercato, "
                        "Beta<1 meno volatile, Beta>1 amplifica i movimenti."
                    ),
                )
                col_c.metric(
                    "Correlazione",
                    f"{ab['correlation']:.2f}",
                    help="Correlazione tra rendimenti giornalieri del portafoglio e del benchmark.",
                )
            except ValueError as e:
                st.warning(f"⚠️ Impossibile calcolare alpha/beta: {e}")

    # ============================================================
    # GRAFICO PERFORMANCE
    # ============================================================
    st.header("📈 Performance storica")

    value = portfolio_cumulative_value(prices.dropna(), weights, initial_value=100)

    # Costruiamo un DataFrame con Portfolio (+ Benchmark se presente)
    performance_df = pd.DataFrame({"Portfolio": value})

    if benchmark_prices is not None:
        # Normalizziamo il benchmark partendo da 100 sulla stessa data iniziale del portafoglio
        bench_series = benchmark_prices[benchmark_ticker]
        bench_aligned = bench_series.reindex(value.index).ffill().dropna()
        bench_normalized = bench_aligned / bench_aligned.iloc[0] * 100
        performance_df[f"Benchmark ({benchmark_ticker})"] = bench_normalized

    performance_df_long = performance_df.reset_index().melt(
        id_vars="Date", var_name="Serie", value_name="Valore (€)"
    )

    fig = px.line(
        performance_df_long,
        x="Date",
        y="Valore (€)",
        color="Serie",
        title="Valore del portafoglio vs benchmark (base = €100)",
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