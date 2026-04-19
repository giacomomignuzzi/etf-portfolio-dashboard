"""
portfolio.py
Costruzione e analisi di portafogli di ETF con pesi custom.
"""

import numpy as np
import pandas as pd

from src.metrics import daily_returns

def validate_weights(tickers: list[str], weights: list[float]) -> np.ndarray:
    """
    Valida che i pesi siano coerenti con i ticker e sommino a 1.

    Args:
        tickers: lista dei ticker.
        weights: lista dei pesi corrispondenti.

    Returns:
        Array numpy con i pesi validati.

    Raises:
        ValueError: se il numero non coincide o se la somma non è 1.
    """
    if len(tickers) != len(weights):
        raise ValueError(
            f"Numero di ticker ({len(tickers)}) diverso "
            f"dal numero di pesi ({len(weights)})."
        )

    weights_array = np.array(weights, dtype=float)

    if not np.isclose(weights_array.sum(), 1.0, atol=1e-4):
        raise ValueError(
            f"I pesi devono sommare a 1, ma sommano a {weights_array.sum():.4f}. "
            f"Pesi ricevuti: {weights}"
        )

    if (weights_array < 0).any():
        raise ValueError(
            f"I pesi non possono essere negativi. Pesi ricevuti: {weights}"
        )

    return weights_array


def portfolio_returns(
    prices: pd.DataFrame,
    weights: list[float],
    rebalance: bool = True,
) -> pd.Series:
    """
    Calcola i rendimenti giornalieri del portafoglio.

    Args:
        prices: DataFrame con i prezzi (colonne = ticker).
        weights: lista di pesi nello stesso ordine delle colonne di prices.
        rebalance: se True, i pesi sono ribilanciati ogni giorno (constant-mix).
                   se False, strategia buy & hold (i pesi driftano nel tempo).

    Returns:
        Series con i rendimenti giornalieri del portafoglio.
    """
    w = validate_weights(list(prices.columns), weights)

    returns = daily_returns(prices)

    if rebalance:
        # Constant-mix: ogni giorno il portafoglio viene ribilanciato ai pesi target
        # r_p = sum(w_i * r_i)
        portfolio_r = (returns * w).sum(axis=1)
    else:
        # Buy & hold: calcoliamo il valore del portafoglio e poi i rendimenti
        portfolio_value = portfolio_cumulative_value(prices, weights)
        portfolio_r = portfolio_value.pct_change().dropna()

    return portfolio_r


def portfolio_cumulative_value(
    prices: pd.DataFrame,
    weights: list[float],
    initial_value: float = 100.0,
) -> pd.Series:
    """
    Simula il valore di un portafoglio buy & hold nel tempo.

    Args:
        prices: DataFrame con i prezzi.
        weights: pesi iniziali.
        initial_value: valore iniziale del portafoglio (default 100).

    Returns:
        Series con il valore del portafoglio giorno per giorno.
    """
    w = validate_weights(list(prices.columns), weights)

    # Allocazione iniziale in valore (es. 60€ su VWCE, 40€ su AGGH se weights=[0.6, 0.4])
    initial_allocation = initial_value * w

    # Numero di "quote" acquistate per ogni asset al giorno 0
    shares = initial_allocation / prices.iloc[0]

    # Valore del portafoglio ogni giorno = somma (quote * prezzo corrente)
    portfolio_value = (prices * shares).sum(axis=1)

    return portfolio_value


def portfolio_cumulative_returns(
    prices: pd.DataFrame,
    weights: list[float],
    rebalance: bool = True,
) -> pd.Series:
    """
    Calcola il rendimento cumulato del portafoglio.

    Args:
        prices: DataFrame con i prezzi.
        weights: pesi.
        rebalance: strategia di ribilanciamento.

    Returns:
        Series con il rendimento cumulato (0 al primo giorno).
    """
    returns = portfolio_returns(prices, weights, rebalance=rebalance)
    return (1 + returns).cumprod() - 1


def correlation_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola la matrice di correlazione dei rendimenti giornalieri.

    Args:
        prices: DataFrame con i prezzi.

    Returns:
        DataFrame quadrato con le correlazioni tra asset.
    """
    returns = daily_returns(prices)
    return returns.corr()

def portfolio_summary(
    prices: pd.DataFrame,
    weights: list[float],
    risk_free_rate: float = 0.0,
    rebalance: bool = True,
) -> pd.Series:
    """
    Calcola tutte le metriche del portafoglio in un colpo solo.

    Args:
        prices: DataFrame con i prezzi.
        weights: pesi del portafoglio.
        risk_free_rate: tasso risk-free annualizzato.
        rebalance: True = constant-mix, False = buy & hold.

    Returns:
        Series con CAGR, Volatility, Sharpe Ratio, Max Drawdown del portafoglio.
    """
    # Rimuoviamo righe con valori mancanti per evitare distorsioni
    clean_prices = prices.dropna()

    # Rendimenti del portafoglio
    port_returns = portfolio_returns(clean_prices, weights, rebalance=rebalance)

    # Valore cumulato (partendo da 1)
    cum_value = (1 + port_returns).cumprod()

    # Metriche calcolate direttamente sui rendimenti
    n_days = (cum_value.index[-1] - cum_value.index[0]).days
    n_years = n_days / 365.25

    cagr_val = cum_value.iloc[-1] ** (1 / n_years) - 1
    vol_val = port_returns.std() * np.sqrt(252)

    rf_daily = (1 + risk_free_rate) ** (1 / 252) - 1
    sharpe_val = (port_returns.mean() - rf_daily) * 252 / vol_val

    # Max drawdown sul valore cumulato
    running_max = cum_value.cummax()
    dd = (cum_value - running_max) / running_max
    mdd_val = dd.min()

    return pd.Series({
        "CAGR": cagr_val,
        "Volatility": vol_val,
        "Sharpe Ratio": sharpe_val,
        "Max Drawdown": mdd_val,
    })

def portfolio_ter(
    weights: list[float],
    ter_per_asset: list[float],
) -> float:
    """
    Calcola il TER medio ponderato del portafoglio.

    Args:
        weights: pesi del portafoglio (devono sommare a 1).
        ter_per_asset: TER di ciascun asset, espresso in decimali
                       (es. 0.0022 per un TER dello 0.22%).

    Returns:
        TER medio ponderato (in decimali).
        Es: 0.0015 = 0.15% annuo.

    Raises:
        ValueError: se i due input hanno lunghezze diverse o TER negativi.
    """
    if len(weights) != len(ter_per_asset):
        raise ValueError(
            f"Numero pesi ({len(weights)}) diverso dal numero di TER "
            f"({len(ter_per_asset)})."
        )

    if any(t < 0 for t in ter_per_asset):
        raise ValueError("I TER non possono essere negativi.")

    # Media pesata: sum(w_i * ter_i)
    weighted_ter = sum(w * t for w, t in zip(weights, ter_per_asset))
    return weighted_ter

if __name__ == "__main__":
    from src.data_loader import download_prices
    from src.metrics import summary_metrics

    # Portfolio 60/40 classico
    tickers = ["VWCE.DE", "AGGH.MI"]
    weights = [0.6, 0.4]

    print(f"Portfolio: {dict(zip(tickers, weights))}")
    prices = download_prices(tickers, "2020-01-01", "2024-12-31")

    # Simula valore portafoglio (partendo da 100€)
    value = portfolio_cumulative_value(prices, weights, initial_value=100)
    print(f"\n💰 Valore iniziale: €{value.iloc[0]:.2f}")
    print(f"💰 Valore finale:   €{value.iloc[-1]:.2f}")
    print(f"📈 Rendimento totale: {(value.iloc[-1] / value.iloc[0] - 1) * 100:.2f}%")

    # Matrice correlazione
    print(f"\n🔗 Correlazione VWCE-AGGH:")
    print(correlation_matrix(prices).round(3))

    # Metriche del portafoglio (buy & hold vs rebalanced)
    print(f"\n📊 Metriche del portafoglio 60/40:")
    pv_rebal = portfolio_cumulative_value(prices, weights)
    # Trasformiamo il valore portafoglio in "prezzo" per usare le funzioni di metrics
    portfolio_as_prices = pv_rebal.to_frame("Portfolio_60_40")
    print(summary_metrics(portfolio_as_prices, risk_free_rate=0.03))