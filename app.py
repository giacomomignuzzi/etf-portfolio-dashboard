"""
app.py — ETF Portfolio Dashboard
Interfaccia Streamlit per analizzare portafogli di ETF.
"""

import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import numpy as np

from src.data_loader import download_prices
from src.metrics import alpha_beta, daily_returns
from src.reference_data import get_etf_dataframe, get_index_dataframe, detect_currency_from_ticker
from src.portfolio import (
    portfolio_returns,
    portfolio_cumulative_value,
    portfolio_cumulative_returns,
    portfolio_summary,
    correlation_matrix,
    portfolio_ter,
    simulate_dca,
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

# Input TER per ogni ETF (opzionale)
ter_input = st.sidebar.text_input(
    "TER (% annuo, separati da virgola)",
    value="0.22, 0.10",
    help=(
        "Total Expense Ratio di ogni ETF (es. 0.22 per 0.22%). "
        "Trovi questo dato su justetf.com o sul sito del provider."
    ),
)
try:
    ters = [float(t.strip()) / 100 for t in ter_input.split(",") if t.strip()]
except ValueError:
    st.sidebar.error("TER non validi. Inserisci numeri separati da virgole.")
    st.stop()

# Capitale ipotetico
capital_input = st.sidebar.number_input(
    "Capitale investito ipotetico (€)",
    min_value=1000,
    max_value=10_000_000,
    value=10_000,
    step=1000,
    help="Serve solo per calcolare quanto paghi di TER in euro ogni anno.",
)

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
st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ Strategia")

strategy = st.sidebar.radio(
    "Tipo di gestione",
    options=["Rebalanced", "Buy & Hold"],
    index=0,
    help=(
        "**Rebalanced**: ripristina periodicamente i pesi target. "
        "**Buy & Hold**: compra il primo giorno e mantiene fino alla fine "
        "(i pesi driftano nel tempo)."
    ),
)
is_rebalanced = strategy == "Rebalanced"

if is_rebalanced:
    rebalance_frequency = st.sidebar.selectbox(
        "Unità di frequenza",
        options=["daily", "monthly", "quarterly", "yearly"],
        index=3,
        help=(
            "Unità temporale base per il ribilanciamento. "
            "Combinala con 'ogni N' per strategie custom (es: ogni 2 anni)."
        ),
    )

    if rebalance_frequency == "daily":
        rebalance_every_n = 1
        st.sidebar.caption("ℹ️ Daily ignora il parametro 'ogni N'.")
    else:
        rebalance_every_n = st.sidebar.number_input(
            f"Ogni N {rebalance_frequency}",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
            help=(
                f"Ribilancia ogni N {rebalance_frequency}. "
                f"Es: 2 con 'yearly' → ogni 2 anni. "
                f"1 = ogni periodo (comportamento standard)."
            ),
        )

    if rebalance_frequency == "daily":
        strategy_label = "ogni giorno"
    elif rebalance_every_n == 1:
        strategy_label = {
            "monthly": "ogni mese",
            "quarterly": "ogni trimestre",
            "yearly": "ogni anno",
        }[rebalance_frequency]
    else:
        unit_label = {
            "monthly": "mesi",
            "quarterly": "trimestri",
            "yearly": "anni",
        }[rebalance_frequency]
        strategy_label = f"ogni {rebalance_every_n} {unit_label}"

    st.sidebar.caption(f"🔁 Ribilanciamento: **{strategy_label}**.")
else:
    rebalance_frequency = "daily"
    rebalance_every_n = 1

# Bottone di avvio
run_button = st.sidebar.button("🚀 Analizza Portfolio", type="primary")

# ─── SIMULAZIONE PAC (Piano d'Accumulo) ───
st.sidebar.markdown("---")
st.sidebar.subheader("💰 Simulazione PAC")

enable_pac = st.sidebar.checkbox(
    "Simula un Piano d'Accumulo",
    value=False,
    help="Invece di investire tutto in un colpo, versa una cifra fissa ogni mese.",
)

if enable_pac:
    monthly_contribution = st.sidebar.number_input(
        "Versamento mensile (€)",
        min_value=50.0,
        max_value=10000.0,
        value=300.0,
        step=50.0,
        help="Cifra versata il primo giorno di trading di ogni mese.",
    )
    pac_start_date = st.sidebar.date_input(
        "Data inizio PAC",
        value=start_date,
        min_value=start_date,
        max_value=end_date,
        help="Quando parte il primo versamento.",
    )
else:
    monthly_contribution = None
    pac_start_date = None

# Expander con liste di riferimento
st.sidebar.markdown("---")
with st.sidebar.expander("📘 Ticker di riferimento"):
    tab1, tab2 = st.tabs(["ETF", "Indici"])

    with tab1:
        st.markdown("**ETF UCITS comuni** (per investitori EU)")
        etf_df = get_etf_dataframe()
        st.dataframe(etf_df, use_container_width=True, hide_index=True, height=300)
        st.caption(
            "💡 Copia il Ticker e incollalo nel campo 'Ticker' in alto. "
            "Verifica TER aggiornati su justetf.com."
        )

    with tab2:
        st.markdown("**Indici di mercato** (per benchmark)")
        idx_df = get_index_dataframe()
        st.dataframe(idx_df, use_container_width=True, hide_index=True, height=300)
        st.caption(
            "💡 Gli indici non includono TER né dividendi; usa un ETF "
            "corrispondente se vuoi un confronto più realistico."
        )

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

    # =========================================================
    # CHECK VALUTE INTERNO AL PORTFOLIO
    # =========================================================
    if len(tickers) > 1:
        asset_currencies = {t: detect_currency_from_ticker(t) for t in tickers}
        known_currencies = {c for c in asset_currencies.values() if c != "UNKNOWN"}

        if len(known_currencies) > 1:
            detail = ", ".join(
                f"`{ticker}` ({curr})" for ticker, curr in asset_currencies.items()
            )
            st.warning(
                f"⚠️ **Valute miste nel portafoglio**: i tuoi asset sono quotati in "
                f"valute diverse ({detail}). I rendimenti e la volatilità del portafoglio "
                f"riflettono sia l'andamento degli asset sia le oscillazioni del cambio. "
                f"Considera di usare ETF tutti nella stessa valuta (es. tutti EUR) "
                f"oppure versioni EUR-hedged dove disponibili."
            )
    # ============================================================
    # METRICHE PRINCIPALI
    # ============================================================
    st.header("📊 Metriche del portafoglio")

    summary = portfolio_summary(
        prices,
        weights,
        risk_free_rate=risk_free,
        rebalance=is_rebalanced,
        rebalance_frequency=rebalance_frequency,
        rebalance_every_n=rebalance_every_n,
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
    # Currency-awareness check
        portfolio_currencies = {detect_currency_from_ticker(t) for t in tickers}
        benchmark_currency = detect_currency_from_ticker(benchmark_ticker)

        # Rimuovi "UNKNOWN" per valutare solo le valute identificate
        known_portfolio_currencies = portfolio_currencies - {"UNKNOWN"}

        if (
            benchmark_currency != "UNKNOWN"
            and known_portfolio_currencies
            and benchmark_currency not in known_portfolio_currencies
        ):
            portfolio_curr_str = ", ".join(sorted(known_portfolio_currencies))
            st.warning(
                f"⚠️ **Attenzione valute**: il portafoglio è in **{portfolio_curr_str}** "
                f"ma il benchmark `{benchmark_ticker}` è in **{benchmark_currency}**. "
                f"I rendimenti includono effetti di cambio valutario. "
                f"Per un confronto più pulito, usa un benchmark nella stessa valuta "
                f"(vedi 'Ticker di riferimento' nella sidebar)."
            )
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
                prices.dropna(),
                weights,
                rebalance=is_rebalanced,
                rebalance_frequency=rebalance_frequency,
                rebalance_every_n=rebalance_every_n,
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
    # COSTI DEL PORTAFOGLIO (TER)
    # ============================================================
    if len(ters) == len(weights):
        st.header("💸 Costi del portafoglio")

        try:
            port_ter = portfolio_ter(weights, ters)
            annual_cost_eur = port_ter * capital_input

            col_t1, col_t2, col_t3 = st.columns(3)
            col_t1.metric(
                "TER portafoglio (% annuo)",
                f"{port_ter * 100:.3f}%",
                help="Media pesata dei TER dei singoli ETF.",
            )
            col_t2.metric(
                "Costo annuo",
                f"€{annual_cost_eur:,.2f}",
                help=f"Su un capitale di €{capital_input:,.0f}.",
            )
            col_t3.metric(
                "Costo in 10 anni",
                f"€{annual_cost_eur * 10:,.2f}",
                help=(
                    "Stima semplificata: TER costante × 10 anni. "
                    "La stima reale è più alta per via della capitalizzazione "
                    "(il TER si applica ogni anno sul capitale che cresce)."
                ),
            )

            # Dettaglio per asset
            with st.expander("📋 Dettaglio TER per asset"):
                ter_detail = pd.DataFrame({
                    "Ticker": tickers,
                    "Peso": [f"{w * 100:.1f}%" for w in weights],
                    "TER": [f"{t * 100:.3f}%" for t in ters],
                    "Contributo al TER totale": [
                        f"{w * t * 100:.4f}%" for w, t in zip(weights, ters)
                    ],
                })
                st.dataframe(ter_detail, use_container_width=True, hide_index=True)

        except ValueError as e:
            st.warning(f"⚠️ Impossibile calcolare il TER: {e}")
    else:
        st.info(
            f"ℹ️ Hai inserito {len(ters)} TER ma {len(weights)} pesi. "
            "Aggiungi un TER per ogni ETF per vedere i costi."
        )

    # =========================================================
    # SIMULAZIONE PAC
    # =========================================================
    if enable_pac:
        st.header("💰 Simulazione Piano d'Accumulo (PAC)")

        try:
            pac_df = simulate_dca(
                prices=prices,
                weights=weights,
                monthly_contribution=monthly_contribution,
                start_date=pd.Timestamp(pac_start_date),
            )

            total_invested = pac_df["Versato"].iloc[-1]
            final_value = pac_df["Valore"].iloc[-1]
            absolute_gain = final_value - total_invested
            percentage_gain = (absolute_gain / total_invested) * 100 if total_invested > 0 else 0.0

            # Confronto con PIC (Lump Sum): stesso capitale totale investito all'inizio
            pic_prices = prices.loc[prices.index >= pd.Timestamp(pac_start_date)]
            if not pic_prices.empty:
                initial_prices = pic_prices.iloc[0].values
                units_pic = (total_invested * np.asarray(weights)) / initial_prices
                final_prices = pic_prices.iloc[-1].values
                final_value_pic = (units_pic * final_prices).sum()
                pic_gain_pct = ((final_value_pic - total_invested) / total_invested) * 100
            else:
                final_value_pic = total_invested
                pic_gain_pct = 0.0

            # Metriche principali
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("💶 Capitale versato", f"€ {total_invested:,.0f}")
            col2.metric("📈 Valore finale (PAC)", f"€ {final_value:,.0f}",
                        delta=f"{percentage_gain:+.2f}%")
            col3.metric("💼 Valore finale (PIC)", f"€ {final_value_pic:,.0f}",
                        delta=f"{pic_gain_pct:+.2f}%")
            diff = final_value - final_value_pic
            col4.metric("⚖️ PAC vs PIC", f"€ {diff:+,.0f}",
                        delta=f"{percentage_gain - pic_gain_pct:+.2f} p.p.")

            # Spiegazione contestuale
            if final_value > final_value_pic:
                st.success(
                    f"✅ In questo periodo il **PAC ha battuto il PIC** di €{diff:,.0f}. "
                    f"Questo accade in mercati volatili o con drawdown nel mezzo: "
                    f"il PAC ha comprato quote a prezzi più bassi durante i ribassi."
                )
            else:
                st.info(
                    f"ℹ️ In questo periodo il **PIC ha battuto il PAC** di €{-diff:,.0f}. "
                    f"Questo è tipico dei mercati in trend rialzista: "
                    f"investire tutto subito significa esporsi prima alla crescita."
                )

            # Grafico PAC
            fig_pac = go.Figure()
            fig_pac.add_trace(go.Scatter(
                x=pac_df.index, y=pac_df["Versato"],
                mode="lines", name="Capitale versato",
                line=dict(color="gray", dash="dash", width=2),
            ))
            fig_pac.add_trace(go.Scatter(
                x=pac_df.index, y=pac_df["Valore"],
                mode="lines", name="Valore portfolio (PAC)",
                line=dict(color="#2E8B57", width=3),
            ))
            fig_pac.update_layout(
                title="PAC: capitale versato vs valore del portafoglio",
                xaxis_title="Data",
                yaxis_title="Euro (€)",
                hovermode="x unified",
                height=450,
            )
            st.plotly_chart(fig_pac, use_container_width=True)

            # Nota educativa
            with st.expander("📘 PAC vs PIC: quale è meglio?"):
                st.markdown("""
                **PIC (Piano di Investimento del Capitale)**: investi tutto subito.
                Storicamente batte il PAC nel ~65-70% dei periodi perché i mercati salgono
                più spesso di quanto scendano (time in market > timing the market).

                **PAC (Piano d'Accumulo)**: investi gradualmente. Vantaggi:
                - Adatto a chi non ha un capitale iniziale ma uno stipendio mensile
                - Riduce il rischio di "entrare al picco"
                - Impatto psicologico minore nei drawdown (si continua a comprare "a sconto")

                **Quando il PAC batte il PIC?**
                Quando il mercato scende nel mezzo del periodo e poi recupera: il PAC
                compra quote a prezzi più bassi, abbassando il prezzo medio di carico.

                **Dollar-Cost Averaging** è il nome tecnico della strategia PAC in inglese.
                """)

        except ValueError as e:
            st.error(f"❌ Errore nella simulazione PAC: {e}")

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