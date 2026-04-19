"""
main.py - Script di test per verificare l'import dei moduli.
"""

from src.data_loader import download_prices


def main():
    tickers = ["VWCE.DE", "AGGH.MI"]
    prices = download_prices(tickers, start_date="2020-01-01")

    print("=" * 50)
    print("PORTFOLIO DATA LOADED")
    print("=" * 50)
    print(f"Tickers: {tickers}")
    print(f"Periodo: {prices.index.min().date()} → {prices.index.max().date()}")
    print(f"Giorni di trading: {len(prices)}")
    print(f"\nUltimi 5 giorni:")
    print(prices.tail())
    print(f"\nStatistiche sui prezzi:")
    print(prices.describe())


if __name__ == "__main__":
    main()