import datetime
from . order import Order
from . trading_const import Direction, OrderType, Offset
from . trading_object import SubscribeRequest
from . event_engine import PriEventEngine
from . instrument import StockOptionInst, Stock, Future, FutOptionInst

class PseudoAgent(object):
    def __init__(self, config = {}, tday = datetime.date.today()):
        self.instruments = {}
        self.scur_day = tday
        self.tick_id = 0
        self.eod_flag = False
        self.event_engine = self.event_engine = PriEventEngine(0.5)
        self.folder = config.get("folder", "")
        self.instruments = {}

    def add_instrument(self, instID, exch):
        if instID not in self.instruments:
            if exch in ["SSE", "SZSE", "NYSE", "NASDAQ", "HKSE"]:
                self.instruments[instID] = Stock(instID)
            elif exch in ["CFFEX", "SHFE", "DCE", "CZCE", "INE", "GFEX", "NYMEX", "GLOBEX", "COMEX", "ICE", "CME", "CBOT"]:
                self.instruments[instID] = Future(instID)
            self.instruments[instID].update_param(self.scur_day)

def create_order(instID, exch, price, pos, price_type = "Limit", offset = "Open", direction = ""):
    if direction == "Net":
        direction = Direction(direction)
        vol = pos
    else:
        direction = Direction.LONG if pos > 0 else Direction.SHORT
        vol = int(abs(pos))
    iorder = Order(instID = instID, exchange = exch, limit_price = price, volume = vol, \
                action_type = Offset(offset), direction = direction, price_type = OrderType(price_type))
    return iorder

def create_sub_req(instID, exch):
    sub_req = SubscribeRequest(symbol=instID, exchange=exch)
    return sub_req