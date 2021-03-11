from dataclasses import dataclass
import datetime
from logging import INFO
from . trading_const import *
from pycmqlib3.utility.misc import get_tick_id

@dataclass
class BaseData:
    """
    Any data object needs a gateway_name as source
    and should inherit base data.
    """

    gateway_name: str


@dataclass
class TickData(BaseData):
    """
    Tick data contains information about:
        * last trade in market
        * orderbook snapshot
        * intraday market statistics.
    """

    instID: str
    exchange: str
    timestamp: datetime.datetime

    tick_id: int = 0
    volume: float = 0
    openInterest: float = 0
    price: float = 0
    last_vol: float = 0
    up_limit: float = 1e+6
    down_limit: float = -1e+6

    open: float = 0
    high: float = 0
    low: float = 0
    prev_close: float = 0

    bid_price1: float = 0
    bid_price2: float = 0
    bid_price3: float = 0
    bid_price4: float = 0
    bid_price5: float = 0

    ask_price1: float = 0
    ask_price2: float = 0
    ask_price3: float = 0
    ask_price4: float = 0
    ask_price5: float = 0

    bid_vol1: float = 0
    bid_vol2: float = 0
    bid_vol3: float = 0
    bid_vol4: float = 0
    bid_vol5: float = 0

    ask_vol1: float = 0
    ask_vol2: float = 0
    ask_vol3: float = 0
    ask_vol4: float = 0
    ask_vol5: float = 0

    def __post_init__(self):
        """"""
        self.global_inst_key = f"{self.instID}.{self.exchange}"
        self.date = self.timestamp.date()


@dataclass
class BarData(BaseData):
    """
    Candlestick bar data of a certain trading period.
    """

    symbol: str
    exchange: str
    datetime: datetime.datetime

    interval: Interval = None
    volume: float = 0
    openInterest: float = 0
    open: float = 0
    high: float = 0
    low: float = 0
    close: float = 0

    def __post_init__(self):
        """"""
        self.global_inst_key = f"{self.symbol}.{self.exchange}"

@dataclass
class OrderData(BaseData):
    """
    Order data contains information for tracking lastest status
    of a specific order.
    """

    symbol: str
    exchange: str
    orderid: str

    type: OrderType = OrderType.LIMIT
    direction: Direction = None
    offset: Offset = Offset.NONE
    price: float = 0
    volume: float = 0
    traded: float = 0
    status: OrderStatus = OrderStatus.Ready
    datetime: datetime = None

    def __post_init__(self):
        """"""
        self.global_inst_key = f"{self.symbol}.{self.exchange}"
        self.global_orderid = f"{self.gateway_name}.{self.orderid}"

    def is_active(self) -> bool:
        """
        Check if the order is active.
        """
        if self.status in Alive_Order_Status:
            return True
        else:
            return False

    def create_cancel_request(self) -> "CancelRequest":
        """
        Create cancel request object from order.
        """
        req = CancelRequest(
            orderid=self.orderid, symbol=self.symbol, exchange=self.exchange
        )
        return req


@dataclass
class TradeData(BaseData):
    """
    Trade data contains information of a fill of an order. One order
    can have several trade fills.
    """

    symbol: str
    exchange: str
    orderid: str
    tradeid: str
    direction: Direction = None

    offset: Offset = Offset.NONE
    price: float = 0
    volume: float = 0
    datetime: datetime = None

    def __post_init__(self):
        """"""
        self.global_inst_key = f"{self.symbol}.{self.exchange}"
        self.global_orderid = f"{self.gateway_name}.{self.orderid}"
        self.global_tradeid = f"{self.gateway_name}.{self.tradeid}"


@dataclass
class PositionData(BaseData):
    """
    Positon data is used for tracking each individual position holding.
    """

    symbol: str
    exchange: str
    direction: Direction

    volume: float = 0
    frozen: float = 0
    price: float = 0
    pnl: float = 0
    yd_volume: float = 0

    def __post_init__(self):
        """"""
        self.global_inst_key = f"{self.symbol}.{self.exchange}"
        self.vt_positionid = f"{self.global_inst_key}.{self.direction.value}"


@dataclass
class AccountData(BaseData):
    """
    Account data contains information about balance, frozen and
    available.
    """

    accountid: str

    balance: float = 0
    frozen: float = 0

    def __post_init__(self):
        """"""
        self.available = self.balance - self.frozen
        self.vt_accountid = f"{self.gateway_name}.{self.accountid}"


@dataclass
class LogData(BaseData):
    """
    Log data is used for recording log messages on GUI or in log files.
    """

    msg: str
    level: int = INFO

    def __post_init__(self):
        """"""
        self.time = datetime.datetime.now()


@dataclass
class ContractData(BaseData):
    """
    Contract data contains basic information about each contract traded.
    """

    symbol: str
    exchange: str
    name: str
    product: ProductType
    size: int
    pricetick: float

    min_volume: float = 1           # minimum trading volume of the contract
    stop_supported: bool = False    # whether server supports stop order
    net_position: bool = False      # whether gateway uses net position volume
    history_data: bool = False      # whether gateway provides bar history data

    option_strike: float = 0
    option_underlying: str = ""     # global_inst_key of underlying contract
    option_type: OptionType = None
    option_expiry: datetime = None
    option_portfolio: str = ""
    option_index: str = ""          # for identifying options with same strike price

    def __post_init__(self):
        """"""
        self.global_inst_key = f"{self.symbol}.{self.exchange}"


@dataclass
class SubscribeRequest:
    """
    Request sending to specific gateway for subscribing tick data update.
    """

    symbol: str
    exchange: str

    def __post_init__(self):
        """"""
        self.global_inst_key = f"{self.symbol}.{self.exchange}"

@dataclass
class OrderRequest:
    """
    Request sending to specific gateway for creating a new order.
    """

    symbol: str
    exchange: str
    direction: Direction
    type: OrderType
    volume: float
    price: float = 0
    offset: Offset = Offset.NONE
    reference: str = ""

    def __post_init__(self):
        """"""
        self.global_inst_key = f"{self.symbol}.{self.exchange}"

    def create_order_data(self, orderid: str, gateway_name: str) -> OrderData:
        """
        Create order data from request.
        """
        order = OrderData(
            symbol=self.symbol,
            exchange=self.exchange,
            orderid=orderid,
            type=self.type,
            direction=self.direction,
            offset=self.offset,
            price=self.price,
            volume=self.volume,
            gateway_name=gateway_name,
        )
        return order


@dataclass
class CancelRequest:
    """
    Request sending to specific gateway for canceling an existing order.
    """

    orderid: str
    symbol: str
    exchange: str

    def __post_init__(self):
        """"""
        self.global_inst_key = f"{self.symbol}.{self.exchange}"


@dataclass
class HistoryRequest:
    """
    Request sending to specific gateway for querying history data.
    """

    symbol: str
    exchange: str
    start: datetime.datetime
    end: datetime.datetime = None
    interval: Interval = None

    def __post_init__(self):
        """"""
        self.global_inst_key = f"{self.symbol}.{self.exchange}"