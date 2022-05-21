# encoding: UTF-8
import datetime
import logging
from pycmqlib3.utility.misc import is_workday, day_shift, CHN_Holidays
from pycmqlib3.core.trading_const import Exchange
from pycmqlib3.core.event_engine import Event
from pycmqlib3.core.event_type import EVENT_MARKETDATA, EVENT_DAYSWITCH, \
    EVENT_RTNORDER, EVENT_RTNTRADE, EVENT_ERRORDERINSERT, EVENT_ERRORDERCANCEL,\
    EVENT_WIND_CONNECTREQ
from . ctp_gateway import CtpGateway

w = None

try:
    from WindPy import w
except ImportError:
    print('Need to install WindPy first!')

# 交易所类型映射
exchangeMap = {}
exchangeMap[Exchange.SSE.value] = 'SH'
exchangeMap[Exchange.SZSE.value] = 'SZ'
exchangeMap[Exchange.CFFEX.value] = 'CFE'
exchangeMap[Exchange.SHFE.value] = 'SHF'
exchangeMap[Exchange.DCE.value] = 'DCE'
exchangeMap[Exchange.CZCE.value] = 'CZC'
exchangeMapReverse = {v: k for k, v in list(exchangeMap.items())}

wsqParamMap = {}
wsqParamMap['rt_last'] = 'LastPrice'
wsqParamMap['rt_last_vol'] = 'Volume'
wsqParamMap['rt_oi'] = 'OpenInterest'

wsqParamMap['rt_open'] = 'OpenPrice'
wsqParamMap['rt_high'] = 'HighestPrice'
wsqParamMap['rt_low'] = 'LowestPrice'
wsqParamMap['rt_pre_close'] = 'PreClosePrice'

wsqParamMap['rt_high_limit'] = 'UpperLimitPrice'
wsqParamMap['rt_low_limit'] = 'LowerLimitPrice'

wsqParamMap['rt_bid1'] = 'BidPrice1'
#wsqParamMap['rt_bid2'] = 'BidPrice2'
#wsqParamMap['rt_bid3'] = 'BidPrice3'
#wsqParamMap['rt_bid4'] = 'BidPrice4'
#wsqParamMap['rt_bid5'] = 'BidPrice5'

wsqParamMap['rt_ask1'] = 'AskPrice1'
#wsqParamMap['rt_ask2'] = 'AskPrice2'
#wsqParamMap['rt_ask3'] = 'AskPrice3'
#wsqParamMap['rt_ask4'] = 'AskPrice4'
#wsqParamMap['rt_ask5'] = 'AskPrice5'

wsqParamMap['rt_bsize1'] = 'BidVolume1'
#wsqParamMap['rt_bsize2'] = 'BidVolume2'
#wsqParamMap['rt_bsize3'] = 'BidVolume3'
#wsqParamMap['rt_bsize4'] = 'BidVolume4'
#wsqParamMap['rt_bsize5'] = 'BidVolume5'

wsqParamMap['rt_asize1'] = 'AskVolume1'
#wsqParamMap['rt_asize2'] = 'AskVolume2'
#wsqParamMap['rt_asize3'] = 'AskVolume3'
#wsqParamMap['rt_asize4'] = 'AskVolume4'
#wsqParamMap['rt_asize5'] = 'AskVolume5'

wsqParam = ','.join(list(wsqParamMap.keys()))

class WindCtpSimGateway(CtpGateway):
    def __init__(self, agent, gateway_name='WindSim'):
        super(WindCtpSimGateway, self).__init__(agent, gateway_name, md_api = "ctp.windsim_gateway.WindMdApi", td_api = "ctp.ctpsim_gateway.SimctpTdApi")
        self.qry_enabled = False

    def connect(self):
        self.mdApi.connect()
        self.tdApi.connect()

    def register_event_handler(self):
        self.event_engine.register(EVENT_MARKETDATA + self.gateway_name, self.rsp_market_data)
        self.event_engine.register(EVENT_ERRORDERCANCEL + self.gateway_name, self.err_order_insert)
        self.event_engine.register(EVENT_ERRORDERINSERT + self.gateway_name, self.err_order_action)
        self.event_engine.register(EVENT_RTNTRADE + self.gateway_name, self.rtn_trade)
        self.event_engine.register(EVENT_WIND_CONNECTREQ, self.wind_connect)

    def wind_connect(self, event):
        self.mdApi.wConnect(event)

    def rsp_td_login(self, event):
        pass


class WindMdApi(object):
    def __init__(self, gateway):
        self.gateway = gateway
        self.gateway_name = gateway.gateway_name  # gateway对象名称
        self.w = w
        self.connect_status = False  # 连接状态
        self.reqID = 0
        self.login_status = False  # 登录状态
        self.tick_dict = {}
        self.tick_counter = {}
        self.subscribe_buffer_dict = {}
        self.trading_day = 0

    def connect(self):
        event = Event(type = EVENT_WIND_CONNECTREQ)
        self.gateway.event_engine.put(event)

    def subscribe(self, subscribeReq):
        windSymbol = '.'.join([subscribeReq.symbol.upper(), exchangeMap[subscribeReq.exchange]])
        if self.connect_status:
            data = self.w.wsq(windSymbol, wsqParam, func=self.wsqCallBack)
        else:
            self.subscribe_buffer_dict[windSymbol] = subscribeReq

    def close(self):
        if self.w:
            self.w.stop()

    def wsqCallBack(self, data):
        windSymbol = data.Codes[0]
        if data.ErrorCode < 0:
            print(data.Times[0], data.Data[0])
            return
        if windSymbol in self.tick_dict:
            tick_data = self.tick_dict[windSymbol]
        else:
            tick_data = {}
            symbolSplit = windSymbol.split('.')
            tick_data['InstrumentID'] = str(symbolSplit[0])
            tick_data['ExchangeID'] = exchangeMapReverse[symbolSplit[1]]
            if tick_data['ExchangeID'] in ['DCE', 'SHFE']:
                tick_data['InstrumentID'] = tick_data['InstrumentID'].lower()
            tick_data['timestamp'] = datetime.datetime(2018,1,1,0,0,0)
            tick_data['Volume'] = 0
            self.tick_dict[windSymbol] = tick_data
            self.tick_counter[windSymbol] = 0
        dt = data.Times[0]
        if (is_workday(dt.date(), "CHN") == False) and (dt.time() >= datetime.time(2,30,0)):
            self.gateway.on_log("outside market hours", level = logging.WARNING)
            return
        #if tick_data['timestamp'] >= dt:
        #    self.tick_counter[windSymbol] = max(self.tick_counter[windSymbol] + 1, 8)
        #else:
        self.tick_counter[windSymbol] = 0
        tick_data['timestamp'] = dt
        tick_data['TradingDay'] = dt.date()
        tick_data['UpdateTime'] = dt.time()
        if tick_data['UpdateTime'] > datetime.time(20,56,0):
            tick_data['TradingDay'] = day_shift(tick_data['TradingDay'], '1b', CHN_Holidays)
            if tick_data['TradingDay'] > self.gateway.agent.scur_day:
                event = Event(type=EVENT_DAYSWITCH)
                event.dict['log'] = '换日: %s -> %s' % (self.gateway.agent.scur_day, tick_data['TradingDay'])
                event.dict['date'] = tick_data['TradingDay']
                self.gateway.event_engine.put(event)
        tick_data['TradingDay'] = tick_data['TradingDay'].strftime('%Y%m%d')
        self.trading_day = int(tick_data['TradingDay'])
        tick_data['UpdateTime'] = tick_data['UpdateTime'].strftime('%H:%M:%S')
        tick_data['UpdateMillisec'] = str(self.tick_counter[windSymbol]) + '00'
        fields = data.Fields
        values = data.Data
        for n, field in enumerate(fields):
            field = field.lower()
            key = wsqParamMap[field]
            value = values[n][0]
            if key in ['Volume']:
                tick_data[key] += value
            else:
                tick_data[key] = value
        event = Event(type = EVENT_MARKETDATA + self.gateway_name)
        event.dict['data'] = tick_data
        event.dict['gateway'] = self.gateway_name
        self.gateway.event_engine.put(event)

    def wConnect(self, event):
        """利用事件处理线程去异步连接Wind接口"""
        result = self.w.start()
        if not result.ErrorCode:
            logContent = 'Wind is connected sucessfully'
            log_level = logging.INFO
            self.connect_status = True
            for req in list(self.subscribe_buffer_dict.values()):
                self.subscribe(req)
            self.subscribe_buffer_dict.clear()
        else:
            logContent = 'Wind is failed to connect with error code = %d' % result.ErrorCode
            log_level = logging.WARNING
        self.gateway.on_log(logContent, level = log_level)