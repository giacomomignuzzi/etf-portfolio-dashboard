"""
data_loader.py
Modulo per scaricare prezzi storici di ETF da Yahoo Finance.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime


def download_prices(
    tickers: list[str],
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Scarica i prezzi di chiusura rettificati (Adj Close) di uno o più ticker.

    Args:
        tickers: lista di ticker (es. ["VWCE.DE", "AGGH.MI"])
        start_date: data di inizio nel formato "YYYY-MM-DD"
        end_date: data di fine nel formato "YYYY-MM-DD". Se None, usa oggi.

    Returns:
        DataFrame con colonne = ticker, righe = date, valori = prezzi di chiusura.

    Raises:
        ValueError: se tickers è vuoto o se non viene trovato alcun dato.
    """
    # Validazione input
    if not tickers:
        raise ValueError("La lista dei ticker non può essere vuota.")

    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    # Scarica i dati grezzi da Yahoo Finance
    # auto_adjust=True -> i prezzi Close sono già rettificati per split e dividendi
    raw_data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    # Se non trova niente, yfinance restituisce un DataFrame vuoto
    if raw_data.empty:
        raise ValueError(
            f"Nessun dato trovato per i ticker {tickers} "
            f"nel periodo {start_date} - {end_date}."
        )

    # Estrai solo i prezzi di chiusura (Close)
    # Struttura di raw_data: MultiIndex con (OHLCV, Ticker) come colonne
    if len(tickers) == 1:
        # Caso singolo ticker: raw_data ha colonne semplici
        prices = raw_data[["Close"]].copy()
        prices.columns = tickers  # rinomina "Close" con il nome del ticker
    else:
        # Caso più ticker: selezioniamo solo il livello "Close"
        prices = raw_data["Close"].copy()

    # Rimuovi eventuali righe con tutti NaN (giorni senza dati)
    prices = prices.dropna(how="all")

    return prices


if __name__ == "__main__":
    # Blocco di test: viene eseguito solo se lanci direttamente questo file
    # (non quando viene importato come modulo)
    print("Test download singolo ticker:")
    df_single = download_prices(["VWCE.DE"], start_date="2023-01-01")
    print(df_single.tail())
    print(f"Numero righe: {len(df_single)}\n")

    print("Test download multipli ticker:")
    df_multi = download_prices(
        ["VWCE.DE", "AGGH.MI"],
        start_date="2023-01-01",
        end_date="2024-12-31",
    )
    print(df_multi.tail())
    print(f"Numero righe: {len(df_multi)}")