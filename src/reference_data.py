"""
reference_data.py
Liste di riferimento di ETF e indici più comuni per investitori europei.
Dati curati manualmente al 2025, verificare TER e ticker su justetf.com / Yahoo Finance.
"""

import pandas as pd


# Lista di ETF popolari per investitori europei (UCITS)
# TER in percentuale annua
ETF_REFERENCE = [
    # === AZIONARIO GLOBALE ===
    {"Nome": "Vanguard FTSE All-World", "Ticker": "VWCE.DE", "Asset Class": "Equity Global", "Valuta": "EUR", "TER (%)": 0.22, "Replica": "Fisica"},
    {"Nome": "iShares MSCI World", "Ticker": "SWDA.MI", "Asset Class": "Equity Global", "Valuta": "EUR", "TER (%)": 0.20, "Replica": "Fisica"},
    {"Nome": "iShares Core MSCI World", "Ticker": "IWDA.AS", "Asset Class": "Equity Global", "Valuta": "EUR", "TER (%)": 0.20, "Replica": "Fisica"},
    {"Nome": "SPDR MSCI ACWI IMI", "Ticker": "SPYI.DE", "Asset Class": "Equity Global", "Valuta": "EUR", "TER (%)": 0.17, "Replica": "Fisica"},

    # === AZIONARIO USA ===
    {"Nome": "iShares Core S&P 500", "Ticker": "CSPX.MI", "Asset Class": "Equity USA", "Valuta": "EUR", "TER (%)": 0.07, "Replica": "Fisica"},
    {"Nome": "Vanguard S&P 500", "Ticker": "VUAA.DE", "Asset Class": "Equity USA", "Valuta": "EUR", "TER (%)": 0.07, "Replica": "Fisica"},
    {"Nome": "Invesco Nasdaq-100", "Ticker": "EQQQ.MI", "Asset Class": "Equity USA Tech", "Valuta": "EUR", "TER (%)": 0.30, "Replica": "Fisica"},

    # === AZIONARIO EUROPA ===
    {"Nome": "Amundi Stoxx Europe 600", "Ticker": "MEUD.PA", "Asset Class": "Equity Europe", "Valuta": "EUR", "TER (%)": 0.07, "Replica": "Fisica"},
    {"Nome": "iShares Core MSCI Europe", "Ticker": "IMEU.MI", "Asset Class": "Equity Europe", "Valuta": "EUR", "TER (%)": 0.12, "Replica": "Fisica"},

    # === AZIONARIO EMERGING MARKETS ===
    {"Nome": "iShares Core MSCI EM IMI", "Ticker": "EIMI.MI", "Asset Class": "Equity EM", "Valuta": "EUR", "TER (%)": 0.18, "Replica": "Fisica"},
    {"Nome": "Vanguard FTSE Emerging Markets", "Ticker": "VFEM.DE", "Asset Class": "Equity EM", "Valuta": "EUR", "TER (%)": 0.22, "Replica": "Fisica"},

    # === OBBLIGAZIONARIO ===
    {"Nome": "iShares Global Aggregate Bond EUR Hedged", "Ticker": "AGGH.MI", "Asset Class": "Bond Global Agg", "Valuta": "EUR", "TER (%)": 0.10, "Replica": "Fisica"},
    {"Nome": "iShares Core Euro Government Bond", "Ticker": "IEGA.MI", "Asset Class": "Bond Euro Gov", "Valuta": "EUR", "TER (%)": 0.09, "Replica": "Fisica"},
    {"Nome": "iShares $ Treasury Bond 7-10", "Ticker": "IBTM.MI", "Asset Class": "Bond US Treasury", "Valuta": "EUR", "TER (%)": 0.07, "Replica": "Fisica"},
    {"Nome": "Xtrackers II EUR Corporate Bond", "Ticker": "XBLC.DE", "Asset Class": "Bond Euro Corp", "Valuta": "EUR", "TER (%)": 0.12, "Replica": "Fisica"},

    # === COMMODITIES / GOLD ===
    {"Nome": "Invesco Physical Gold", "Ticker": "SGLD.MI", "Asset Class": "Gold", "Valuta": "USD", "TER (%)": 0.12, "Replica": "Fisica"},
    {"Nome": "iShares Physical Gold", "Ticker": "SGLN.L", "Asset Class": "Gold", "Valuta": "USD", "TER (%)": 0.12, "Replica": "Fisica"},

    # === REIT ===
    {"Nome": "iShares Developed Markets Property", "Ticker": "IWDP.L", "Asset Class": "REIT Global", "Valuta": "USD", "TER (%)": 0.59, "Replica": "Fisica"},
]


# Lista di indici di mercato (Yahoo Finance)
INDEX_REFERENCE = [
    # === USA ===
    {"Nome": "S&P 500", "Ticker": "^GSPC", "Area": "USA", "Valuta": "USD", "Descrizione": "500 maggiori aziende USA"},
    {"Nome": "Nasdaq Composite", "Ticker": "^IXIC", "Area": "USA", "Valuta": "USD", "Descrizione": "Nasdaq (tecnologia USA)"},
    {"Nome": "Dow Jones Industrial", "Ticker": "^DJI", "Area": "USA", "Valuta": "USD", "Descrizione": "30 large cap USA"},
    {"Nome": "Russell 2000", "Ticker": "^RUT", "Area": "USA", "Valuta": "USD", "Descrizione": "Small cap USA"},

    # === EUROPA ===
    {"Nome": "Euro Stoxx 50", "Ticker": "^STOXX50E", "Area": "Eurozona", "Valuta": "EUR", "Descrizione": "50 large cap Eurozona"},
    {"Nome": "Stoxx Europe 600", "Ticker": "^STOXX", "Area": "Europa", "Valuta": "EUR", "Descrizione": "600 aziende europee"},
    {"Nome": "FTSE MIB", "Ticker": "FTSEMIB.MI", "Area": "Italia", "Valuta": "EUR", "Descrizione": "40 blue chip italiane"},
    {"Nome": "DAX 40", "Ticker": "^GDAXI", "Area": "Germania", "Valuta": "EUR", "Descrizione": "40 large cap tedesche"},
    {"Nome": "CAC 40", "Ticker": "^FCHI", "Area": "Francia", "Valuta": "EUR", "Descrizione": "40 large cap francesi"},
    {"Nome": "FTSE 100", "Ticker": "^FTSE", "Area": "Regno Unito", "Valuta": "GBP", "Descrizione": "100 large cap UK"},

    # === ASIA ===
    {"Nome": "Nikkei 225", "Ticker": "^N225", "Area": "Giappone", "Valuta": "JPY", "Descrizione": "225 aziende giapponesi"},
    {"Nome": "Hang Seng", "Ticker": "^HSI", "Area": "Hong Kong", "Valuta": "HKD", "Descrizione": "Blue chip Hong Kong"},

    # === GLOBALE ===
    {"Nome": "MSCI World (via ETF URTH)", "Ticker": "URTH", "Area": "Globale DM", "Valuta": "USD", "Descrizione": "Azionario mondiale sviluppato"},
    {"Nome": "MSCI ACWI (via ETF ACWI)", "Ticker": "ACWI", "Area": "Globale", "Valuta": "USD", "Descrizione": "Globale (sviluppati + emergenti)"},
]


def get_etf_dataframe() -> pd.DataFrame:
    """Restituisce gli ETF di riferimento come DataFrame pandas."""
    return pd.DataFrame(ETF_REFERENCE)


def get_index_dataframe() -> pd.DataFrame:
    """Restituisce gli indici di riferimento come DataFrame pandas."""
    return pd.DataFrame(INDEX_REFERENCE)


def detect_currency_from_ticker(ticker: str) -> str:
    """
    Indovina la valuta di un ticker dal suffisso.

    Args:
        ticker: simbolo Yahoo Finance (es. "VWCE.DE", "^GSPC", "AAPL")

    Returns:
        Stringa con la valuta ("EUR", "USD", "GBP", ecc.) o "UNKNOWN".
    """
    ticker_upper = ticker.upper()

    # Suffissi Yahoo Finance più comuni
    suffix_to_currency = {
        ".DE": "EUR",   # Xetra
        ".MI": "EUR",   # Borsa Italiana
        ".AS": "EUR",   # Amsterdam
        ".PA": "EUR",   # Parigi
        ".MC": "EUR",   # Madrid
        ".BR": "EUR",   # Bruxelles
        ".L": "GBP",    # Londra
        ".TO": "CAD",   # Toronto
        ".T": "JPY",    # Tokyo
        ".HK": "HKD",   # Hong Kong
        ".SW": "CHF",   # Svizzera
    }

    # Indici famosi (iniziano con ^)
    index_currency = {
        "^GSPC": "USD", "^IXIC": "USD", "^DJI": "USD", "^RUT": "USD",
        "^STOXX50E": "EUR", "^STOXX": "EUR", "^GDAXI": "EUR", "^FCHI": "EUR",
        "^FTSE": "GBP",
        "^N225": "JPY", "^HSI": "HKD",
    }

    if ticker_upper in index_currency:
        return index_currency[ticker_upper]

    for suffix, currency in suffix_to_currency.items():
        if ticker_upper.endswith(suffix):
            return suffix_to_currency[suffix]

    # Se non ha suffisso e non è un indice conosciuto: probabilmente USA = USD
    if "." not in ticker and not ticker.startswith("^"):
        return "USD"

    return "UNKNOWN"