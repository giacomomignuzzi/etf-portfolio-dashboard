"""
metrics.py
Calcolo di metriche finanziarie per portafogli: rendimenti, volatilità,
Sharpe ratio, drawdown.
"""

import numpy as np
import pandas as pd

# Numero convenzionale di giorni di trading in un anno
TRADING_DAYS_PER_YEAR = 252


def daily_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Calcola i rendimenti giornalieri semplici: r_t = (P_t / P_{t-1}) - 1.

    Args:
        prices: DataFrame (o Series) con i prezzi; indice = date.

    Returns:
        DataFrame (o Series) con i rendimenti giornalieri.
        La prima riga sarà NaN (non esiste P_{t-1} per il primo giorno).
    """
    return prices.pct_change().dropna(how="all")


def cumulative_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Calcola il rendimento cumulato a partire dal primo giorno.
    Utile per il grafico di performance.

    Args:
        prices: DataFrame (o Series) con i prezzi.

    Returns:
        DataFrame (o Series) con il rendimento cumulato.
        Il primo giorno vale 0 (nessun rendimento accumulato).
    """
    returns = daily_returns(prices)
    return (1 + returns).cumprod() - 1


def cagr(prices: pd.DataFrame | pd.Series) -> pd.Series | float:
    """
    Compound Annual Growth Rate: rendimento annualizzato geometrico.
    Formula: (P_finale / P_iniziale)^(1/n_anni) - 1

    Args:
        prices: DataFrame (o Series) con i prezzi.

    Returns:
        Series (un CAGR per colonna) o float (se input è Series).
    """
    # Numero di giorni effettivi tra prima e ultima data
    n_days = (prices.index[-1] - prices.index[0]).days
    n_years = n_days / 365.25  # teniamo conto degli anni bisestili

    total_return = prices.iloc[-1] / prices.iloc[0]
    return total_return ** (1 / n_years) - 1


def annualized_volatility(
    prices: pd.DataFrame | pd.Series,
) -> pd.Series | float:
    """
    Volatilità annualizzata: deviazione standard dei rendimenti giornalieri
    scalata per radice di 252.

    Args:
        prices: DataFrame (o Series) con i prezzi.

    Returns:
        Series (volatilità per colonna) o float (se input è Series).
    """
    returns = daily_returns(prices)
    return returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(
    prices: pd.DataFrame | pd.Series,
    risk_free_rate: float = 0.0,
) -> pd.Series | float:
    """
    Sharpe ratio annualizzato: (rendimento medio - risk-free) / volatilità.

    Args:
        prices: DataFrame (o Series) con i prezzi.
        risk_free_rate: tasso risk-free annualizzato (es. 0.03 per 3%).
            Default: 0 (ignora il risk-free).

    Returns:
        Series (Sharpe per colonna) o float.
    """
    returns = daily_returns(prices)
    # Converto il tasso annuale in giornaliero
    rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1

    excess_returns = returns - rf_daily
    mean_excess = excess_returns.mean() * TRADING_DAYS_PER_YEAR  # annualizzato
    vol_annualized = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    return mean_excess / vol_annualized


def max_drawdown(prices: pd.DataFrame | pd.Series) -> pd.Series | float:
    """
    Massimo drawdown: peggior perdita da un picco al successivo minimo.
    Restituito come valore negativo (es. -0.35 significa -35%).

    Args:
        prices: DataFrame (o Series) con i prezzi.

    Returns:
        Series (max drawdown per colonna) o float.
    """
    # Running maximum (picco storico a ogni istante)
    running_max = prices.cummax()
    # Drawdown percentuale rispetto al picco
    drawdown = (prices - running_max) / running_max
    # Il massimo drawdown è il minimo (più negativo) della serie
    return drawdown.min()


def summary_metrics(
    prices: pd.DataFrame | pd.Series,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Crea una tabella riassuntiva con tutte le metriche principali.

    Args:
        prices: DataFrame con i prezzi.
        risk_free_rate: tasso risk-free annualizzato.

    Returns:
        DataFrame con righe = metriche, colonne = ticker.
    """
    return pd.DataFrame({
        "CAGR": cagr(prices),
        "Volatility": annualized_volatility(prices),
        "Sharpe Ratio": sharpe_ratio(prices, risk_free_rate),
        "Max Drawdown": max_drawdown(prices),
    }).T  # .T trasposta: così le metriche sono righe e gli asset sono colonne

def alpha_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Calcola alpha e beta del portafoglio rispetto al benchmark (CAPM).

    Args:
        portfolio_returns: rendimenti giornalieri del portafoglio.
        benchmark_returns: rendimenti giornalieri del benchmark.
        risk_free_rate: tasso risk-free annualizzato.

    Returns:
        Dizionario con:
        - 'alpha': rendimento in eccesso annualizzato attribuibile alla gestione attiva
        - 'beta': sensibilità del portafoglio al benchmark
        - 'correlation': correlazione tra portafoglio e benchmark
    """
    # Allineiamo le due serie sulle date comuni
    df = pd.concat(
        [portfolio_returns.rename("portfolio"), benchmark_returns.rename("benchmark")],
        axis=1,
    ).dropna()

    if len(df) < 2:
        raise ValueError(
            "Serie troppo corta per calcolare alpha/beta dopo l'allineamento."
        )

    # Beta = Cov(P, B) / Var(B)
    covariance = df["portfolio"].cov(df["benchmark"])
    variance_benchmark = df["benchmark"].var()
    beta = covariance / variance_benchmark

    # Alpha (CAPM, annualizzato):
    # alpha = R_p - [R_f + beta * (R_b - R_f)]
    rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    mean_portfolio = df["portfolio"].mean()
    mean_benchmark = df["benchmark"].mean()
    alpha_daily = mean_portfolio - (rf_daily + beta * (mean_benchmark - rf_daily))
    alpha_annualized = alpha_daily * TRADING_DAYS_PER_YEAR

    correlation = df["portfolio"].corr(df["benchmark"])

    return {
        "alpha": alpha_annualized,
        "beta": beta,
        "correlation": correlation,
    }

if __name__ == "__main__":
    # Test rapido: importiamo il data_loader e testiamo tutte le metriche
    # NOTA: quando si esegue un modulo dentro src/, l'import relativo funziona solo
    # se lanciato dalla root del progetto come `python -m src.metrics`
    from src.data_loader import download_prices

    print("Scarico dati di test...")
    prices = download_prices(
        ["VWCE.DE", "AGGH.MI"],
        start_date="2020-01-01",
        end_date="2024-12-31",
    )

    print("\n📊 Summary metrics (risk-free = 3%):")
    print(summary_metrics(prices, risk_free_rate=0.03))