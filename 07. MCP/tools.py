from langchain_core.tools import tool
import yfinance as yf


@tool
def get_current_stock_price(ticker: str) -> float:
    """"
    Retorna o valor atual de uma ação com base no código da empresa.
    """
    data = yf.Ticker(ticker)
    return data.fast_info.get("last_price", 0.0)

@tool
def get_history_stock_price(ticker: str, period: str) -> dict:
    """"
    Retorna o histórico do valor de uma ação
    com base no código da empresa e no período.
    period: str
        | Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        | Default is 1mo
        | Can combine with start/end e.g. end = start + period
    """
    data = yf.Ticker(ticker)
    return data.history(period=period).to_dict()

@tool
def get_company_info(ticker: str) -> dict:
    """"
    Retorna dados completos da empresa com base no código da empresa.
    """
    data = yf.Ticker(ticker)
    return data.info