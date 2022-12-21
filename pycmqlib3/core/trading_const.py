from enum import Enum

class Direction(Enum):
    LONG = "Long"
    SHORT = "Short"
    NET = "Net"

def reverse_direction(direction):
    return Direction.SHORT if direction == Direction.LONG else Direction.LONG


class Offset(Enum):
    NONE = "NA"
    OPEN = "Open"
    CLOSE = "Close"
    CLOSETDAY = "CloseTday"
    CLOSEYDAY = "CloseYday"


class OrderStatus(Enum):
    Pending, Ready, Sent, Done, Cancelled = list(range(5))

Alive_Order_Status = [OrderStatus.Ready, OrderStatus.Sent]

class TradeStatus(Enum):
    Pending, Ready, OrderSent, PFilled, Done, Cancelled, StratConfirm = list(range(7))

# Pending: trigger trade, Ready: ok to start process with zero vol, OrderSent: wait for order update
Active_Trade_Status = [TradeStatus.Pending, TradeStatus.Ready, TradeStatus.OrderSent, TradeStatus.PFilled]
Alive_Trade_Status = [TradeStatus.Ready, TradeStatus.OrderSent, TradeStatus.PFilled, TradeStatus.Done]

class ProductType(Enum):
    Equity = "Equity"
    Future = "Future"
    Option = "Option"
    FutOpt = "FutOpt"
    EqOpt = "EqOpt"
    Index = "Index"
    FX = "FX"
    Spot = "Spot"
    ETF = "ETF"
    Bond = "Bond"
    Warrant = "Warrant"
    Spread = "Spread"
    Fund = "Fund"

Option_ProductTypes = [ProductType.FutOpt, ProductType.EqOpt]

class OrderType(Enum):
    LIMIT = "Limit"
    MARKET = "Market"
    STOP = "Stop"
    FAK = "FAK"
    FOK = "FOK"
    RFQ = "RFQ"


class OptionType(Enum):
    CALL = "C"
    PUT = "P"


class Exchange(Enum):
    CFFEX = "CFFEX"         # China Financial Futures Exchange
    SHFE = "SHFE"           # Shanghai Futures Exchange
    CZCE = "CZCE"           # Zhengzhou Commodity Exchange
    DCE = "DCE"             # Dalian Commodity Exchange
    INE = "INE"             # Shanghai International Energy Exchange
    GFEX = "GFEX"             # Guangzhou Future Exchange
    SSE = "SSE"             # Shanghai Stock Exchange
    SZSE = "SZSE"           # Shenzhen Stock Exchange
    SGE = "SGE"             # Shanghai Gold Exchange
    WXE = "WXE"             # Wuxi Steel Exchange
    CFETS = "CFETS"         # China Foreign Exchange Trade System

    # Global
    SMART = "SMART"         # Smart Router for US stocks
    NYSE = "NYSE"           # New York Stock Exchnage
    NASDAQ = "NASDAQ"       # Nasdaq Exchange
    NYMEX = "NYMEX"         # New York Mercantile Exchange
    COMEX = "COMEX"         # a division of theNew York Mercantile Exchange
    GLOBEX = "GLOBEX"       # Globex of CME
    IDEALPRO = "IDEALPRO"   # Forex ECN of Interactive Brokers
    CME = "CME"             # Chicago Mercantile Exchange
    ICE = "ICE"             # Intercontinental Exchange
    SEHK = "SEHK"           # Stock Exchange of Hong Kong
    HKFE = "HKFE"           # Hong Kong Futures Exchange
    HKSE = "HKSE"           # Hong Kong Stock Exchange
    SGX = "SGX"             # Singapore Global Exchange
    CBOT = "CBT"            # Chicago Board of Trade
    CBOE = "CBOE"           # Chicago Board Options Exchange
    CFE = "CFE"             # CBOE Futures Exchange
    DME = "DME"             # Dubai Mercantile Exchange
    EUREX = "EUX"           # Eurex Exchange
    APEX = "APEX"           # Asia Pacific Exchange
    LME = "LME"             # London Metal Exchange
    BMD = "BMD"             # Bursa Malaysia Derivatives
    TOCOM = "TOCOM"         # Tokyo Commodity Exchange
    EUNX = "EUNX"           # Euronext Exchange
    KRX = "KRX"             # Korean Exchange

    OANDA = "OANDA"         # oanda.com

    # CryptoCurrency
    BITMEX = "BITMEX"
    OKEX = "OKEX"
    HUOBI = "HUOBI"
    BITFINEX = "BITFINEX"
    BINANCE = "BINANCE"
    BYBIT = "BYBIT"         # bybit.com
    COINBASE = "COINBASE"
    DERIBIT = "DERIBIT"
    GATEIO = "GATEIO"
    BITSTAMP = "BITSTAMP"

    # Special Function
    LOCAL = "LOCAL"         # For local generated data


class Currency(Enum):
    USD = "USD"
    HKD = "HKD"
    CNY = "CNY"


class Interval(Enum):
    MINUTE = "1m"
    HOUR = "1h"
    DAILY = "d"
    WEEKLY = "w"