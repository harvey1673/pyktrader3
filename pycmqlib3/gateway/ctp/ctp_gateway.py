# -*- coding: utf-8 -*-
import os
import time
import json
from pycmqlib3.utility.base import *
from pycmqlib3.utility.dbaccess import insert_cont_data
from pycmqlib3.utility.misc import get_obj_by_name, get_tick_id, inst2product, trading_hours, check_trading_range
from pycmqlib3.core.gateway import Gateway, GrossGateway
import logging
import datetime
from pycmqlib3.core.position import GrossPosition, SHFEPosition
from pycmqlib3.core.trading_const import OrderStatus, Direction, OrderType, \
    Exchange, Offset, ProductType, OptionType, Alive_Order_Status
from pycmqlib3.core.event_engine import Event
from pycmqlib3.core.event_type import EVENT_MARKETDATA, EVENT_QRYACCOUNT, \
    EVENT_QRYPOSITION, EVENT_QRYORDER, EVENT_QRYTRADE, EVENT_QRYINVESTOR, \
    EVENT_QRYINSTRUMENT, EVENT_TIMER, EVENT_TDLOGIN, EVENT_ETRADEUPDATE, \
    EVENT_RTNORDER, EVENT_RTNTRADE, EVENT_ERRORDERINSERT, EVENT_ERRORDERCANCEL
from pycmqlib3.core.trading_object import TickData
from . ctp_constant import *


STATUS_CTP2CMQ = {
    THOST_FTDC_OAS_Submitted: OrderStatus.Sent,
    THOST_FTDC_OAS_Accepted: OrderStatus.Sent,
    THOST_FTDC_OAS_Rejected: OrderStatus.Cancelled,
    THOST_FTDC_OST_NoTradeQueueing: OrderStatus.Sent,
    THOST_FTDC_OST_PartTradedQueueing: OrderStatus.Sent,
    THOST_FTDC_OST_AllTraded: OrderStatus.Done,
    THOST_FTDC_OST_Canceled: OrderStatus.Cancelled
}

DIRECTION_CMQ2CTP = {
    Direction.LONG: THOST_FTDC_D_Buy,
    Direction.SHORT: THOST_FTDC_D_Sell
}
DIRECTION_CTP2VT = {v: k for k, v in DIRECTION_CMQ2CTP.items()}
DIRECTION_CTP2VT[THOST_FTDC_PD_Long] = Direction.LONG
DIRECTION_CTP2VT[THOST_FTDC_PD_Short] = Direction.SHORT

ORDERTYPE_CMQ2CTP = {
    OrderType.LIMIT: THOST_FTDC_OPT_LimitPrice,
    OrderType.MARKET: THOST_FTDC_OPT_AnyPrice
}
ORDERTYPE_CTP2VT = {v: k for k, v in ORDERTYPE_CMQ2CTP.items()}

OFFSET_CMQ2CTP = {
    Offset.OPEN: THOST_FTDC_OF_Open,
    Offset.CLOSE: THOST_FTDC_OFEN_Close,
    Offset.CLOSETDAY: THOST_FTDC_OFEN_CloseToday,
    Offset.CLOSEYDAY: THOST_FTDC_OFEN_CloseYesterday,
}

OFFSET_CTP2VT = {v: k for k, v in OFFSET_CMQ2CTP.items()}

EXCHANGE_CTP2CMQ = {
    "CFFEX": Exchange.CFFEX,
    "SHFE": Exchange.SHFE,
    "CZCE": Exchange.CZCE,
    "DCE": Exchange.DCE,
    "INE": Exchange.INE
}

PRODUCT_CTP2CMQ = {
    THOST_FTDC_PC_Futures: ProductType.Future,
    THOST_FTDC_PC_Options: ProductType.FutOpt,
    THOST_FTDC_PC_SpotOption: ProductType.EqOpt,
    THOST_FTDC_PC_Combination: ProductType.Spread
}

OPTIONTYPE_CTP2CMQ = {
    THOST_FTDC_CP_CallOptions: OptionType.CALL,
    THOST_FTDC_CP_PutOptions: OptionType.PUT
}

TERT_RESTART = 0 #从本交易日开始重传
TERT_RESUME = 1 #从上次收到的续传
TERT_QUICK = 2 #只传送登录后的流内容

class CtpGateway(GrossGateway):
    """CTP接口"""

    def __init__(self, agent, gateway_name='CTP', \
                    md_api = 'pycmqlib3.gateway.ctp.vnctp_gateway.VnctpMdApi', \
                    td_api = 'pycmqlib3.gateway.ctp.vnctp_gateway.VnctpTdApi'):
        """Constructor"""
        super(CtpGateway, self).__init__(agent, gateway_name)
        self.mdApi = get_obj_by_name(md_api)(self)
        self.tdApi = get_obj_by_name(td_api)(self)
        self.auto_db_update = False
        self.qry_enabled = True         # 是否要启动循环查询
        self.qry_count = 0           # 查询触发倒计时
        self.qry_trigger = 2         # 查询触发点
        self.qry_commands = []
        self.qry_inst_data = {}
        self.system_orders = []
        self.md_data_buffer = 0
        self.td_conn_mode = TERT_QUICK
        self.intraday_close_ratio = {}
        self.cancel_by_qry_order = False
        #self.product_info = 'Zeno'

    def get_pos_class(self, inst):
        pos_args = {}
        if inst.name in self.intraday_close_ratio:
            pos_args['intraday_close_ratio'] = self.intraday_close_ratio[inst.name]
        if inst.exchange == 'SHFE':
            pos_cls = SHFEPosition
        else:
            pos_cls = GrossPosition
        return (pos_cls, pos_args)

    def connect(self):
        """连接"""
        # 载入json文件
        fileName = self.file_prefix + 'connect.json'
        with open(fileName, 'r') as f:
            setting = json.load(f)
        try:
            userID = str(setting['userID'])
            password = str(setting['password'])
            brokerID = str(setting['brokerID'])
            tdAddress = str(setting['tdAddress'])
            mdAddress = str(setting['mdAddress'])
            self.intraday_close_ratio = setting.get('intraday_close_ratio', {})
            self.cancel_by_qry_order = setting.get('intraday_close_ratio', False)
            self.auto_db_update = setting.get('db_cont_update', False)
            #self.product_info = str(setting.get('product_info', 'Zeno'))

        except KeyError:
            self.on_log('连接配置缺少字段，请检查', level = logging.WARNING)
            return            

        if 'authCode' in setting:
            authCode = str(setting['authCode'])
            appID = str(setting['appID'])
        else:
            authCode = None
            appID = None

        # 创建行情和交易接口对象
        self.mdApi.connect(userID, password, brokerID, mdAddress)
        self.tdApi.connect(userID, password, brokerID, tdAddress, authCode, appID)
    
    def write_error(self, msg, error):
        """"""
        error_id = error["ErrorID"]
        error_msg = error["ErrorMsg"]
        msg = f"{msg}，代码：{error_id}，信息：{error_msg}"
        self.on_log(msg, level = logging.ERROR)

    def check_connection(self):
        """检查状态"""        
        qry_status = False
        if not self.mdApi.connect_status:
            self.mdApi.connect()
            qry_status = True
            self.on_log("CTP MD connect_status = False, reconnecting ...", level = logging.WARNING)
        elif not self.mdApi.login_status:
            self.mdApi.login()
            qry_status = True
            self.on_log("CTP MD login_status = False, re-login ...", level = logging.WARNING)
        elif not self.tdApi.connect_status: 
            self.tdApi.connect()
            qry_status = True
            self.on_log("CTP TD connect_status = False, reconnecting ...", level = logging.WARNING)
        elif not self.tdApi.auth_status:
            self.tdApi.authenticate()
            qry_status = True
            self.on_log("CTP TD auth_status = False, re-authenticating ...", level = logging.WARNING)
        elif not self.tdApi.login_status:
            self.tdApi.login()
            qry_status = True
            self.on_log("CTP TD login_status = False, re-logining ...", level = logging.WARNING)
        if qry_status:
            time.sleep(30)
            self.on_log("CTP gateway will check connection in 30 sec.", level = logging.INFO)
            self.qry_commands.append(self.check_connection)
        else:
            self.on_log("CTP gateway TD|MD are both connected.", level = logging.INFO)

    def subscribe(self, subscribeReq):
        self.add_instrument(subscribeReq.symbol)
        self.mdApi.subscribe(subscribeReq)
    
    def send_order(self, iorder):
        """发单"""
        inst = self.agent.instruments[iorder.instrument]
        # 上期所不支持市价单
        if (iorder.price_type == OrderType.MARKET):
            if (iorder.exchange in ['SHFE', 'CFFEX']):
                iorder.price_type = OrderType.LIMIT
                if iorder.direction == Direction.LONG:
                    iorder.limit_price = inst.up_limit
                else:
                    iorder.limit_price = inst.down_limit
                self.on_log('sending limiting local_id=%s inst=%s for SHFE and CFFEX, change to limit order' % (iorder.local_id, inst.name), level = logging.DEBUG)
            else:
                iorder.limit_price = 0.0
        self.tdApi.sendOrder(iorder)
        iorder.status = OrderStatus.Sent
        super(CtpGateway, self).send_order(iorder)
    
    def cancel_order(self, iorder):
        """撤单"""
        self.tdApi.cancelOrder(iorder)
        super(CtpGateway, self).cancel_order(iorder)
    
    def qry_account(self):
        """查询账户资金"""
        self.tdApi.qryAccount()
    
    def qry_position(self):
        """查询持仓"""
        self.tdApi.qryPosition()

    def qry_instrument(self):
        self.tdApi.qryInstrument()

    def qry_trade(self):
        """查询账户资金"""
        self.tdApi.qryTrade()
    
    def qry_order(self):
        """查询持仓"""
        self.tdApi.qryOrder()
    
    def close(self):
        """关闭"""
        self.mdApi.close()
        self.tdApi.close()
    
    def query(self, event):
        """注册到事件处理引擎上的查询函数"""
        if self.qry_enabled:
            self.qry_count += 1
            if self.qry_count < self.qry_trigger:
                return
            self.qry_count = 0
            if len(self.qry_commands)>0:
                self.qry_commands[0]()
                del self.qry_commands[0]

    def register_event_handler(self):
        self.event_engine.register(EVENT_MARKETDATA+self.gateway_name, self.rsp_market_data)
        self.event_engine.register(EVENT_QRYACCOUNT+self.gateway_name, self.rsp_qry_account)
        self.event_engine.register(EVENT_QRYPOSITION+self.gateway_name, self.rsp_qry_position)
        self.event_engine.register(EVENT_QRYTRADE+self.gateway_name, self.rsp_qry_order)
        self.event_engine.register(EVENT_QRYORDER+self.gateway_name, self.rsp_qry_order)
        self.event_engine.register(EVENT_QRYINVESTOR+self.gateway_name, self.rsp_qry_investor)
        self.event_engine.register(EVENT_QRYINSTRUMENT+self.gateway_name, self.rsp_qry_instrument)
        self.event_engine.register(EVENT_ERRORDERINSERT+self.gateway_name, self.err_order_insert)
        self.event_engine.register(EVENT_ERRORDERCANCEL+self.gateway_name, self.err_order_action)
        self.event_engine.register(EVENT_RTNTRADE+self.gateway_name, self.rtn_trade)
        self.event_engine.register(EVENT_RTNORDER+self.gateway_name, self.rtn_order)
        self.event_engine.register(EVENT_TIMER, self.query)
        self.event_engine.register(EVENT_TDLOGIN+self.gateway_name, self.rsp_td_login)

    def rsp_td_login(self, event):
        self.qry_commands.append(self.qry_account)
        self.qry_commands.append(self.qry_position)
        self.qry_commands.append(self.qry_order)
        self.qry_commands.append(self.qry_trade)

    def on_order(self, order):
        pass

    def on_trade(self, trade):
        pass

    def rtn_order(self, event):
        data = event.dict['data']
        local_id = data['OrderRef'].strip()
        if not local_id.isdigit():
            return
        if (local_id not in self.id2order):
            logContent = 'receive order update from other agents, InstID=%s, OrderRef=%s' % (data['InstrumentID'], local_id)
            self.on_log(logContent, level = logging.WARNING)
            return
        myorder = self.id2order[local_id]
        # only update sysID,
        status = myorder.on_order(sys_id = data['OrderSysID'].strip(), price = data['LimitPrice'], volume = 0)
        if data['OrderStatus'] in [THOST_FTDC_OST_Canceled, THOST_FTDC_OST_PartTradedNotQueueing]:
            myorder.on_cancel()
            status = True
        if status:
            event = Event(type=EVENT_ETRADEUPDATE)
            event.dict['trade_ref'] = myorder.trade_ref
            self.event_engine.put(event)

    def rtn_trade(self, event):
        data = event.dict['data']
        local_id = data['OrderRef'].strip()
        if not local_id.isdigit():
            return
        if local_id in self.id2order:
            myorder = self.id2order[local_id]
            myorder.on_trade(price = data['Price'], volume=data['Volume'], trade_id = data['TradeID'])
            event = Event(type=EVENT_ETRADEUPDATE)
            event.dict['trade_ref'] = myorder.trade_ref
            self.event_engine.put(event)
        else:
            logContent = 'receive trade update from other agents, InstID=%s, OrderRef=%s' % (data['InstrumentID'], local_id)
            self.on_log(logContent, level = logging.WARNING)

    def rsp_market_data(self, event):
        data = event.dict['data']
        if self.mdApi.trading_day == 0:
            self.mdApi.trading_day = int(data['TradingDay'])
        timestr = str(self.mdApi.trading_day) + ' '+ str(data['UpdateTime']) + ' ' + str(data['UpdateMillisec']) + '000'
        try:
            timestamp = datetime.datetime.strptime(timestr, '%Y%m%d %H:%M:%S %f')
        except:
            logContent =  "Error to convert timestr = %s" % timestr
            self.on_log(logContent, level = logging.INFO)
            return
        tick_id = get_tick_id(timestamp)
        if data['ExchangeID'] == 'CZCE':
            if (len(data['TradingDay'])>0):
                if (self.mdApi.trading_day > int(data['TradingDay'])) and (tick_id >= 600000):
                    logContent = "tick data is wrong, %s" % data
                    self.on_log(logContent, level = logging.WARNING)
                    return
        tick = TickData(
            instID = data['InstrumentID'],
            exchange = data['ExchangeID'],
            timestamp = timestamp,
            gateway_name = self.gateway_name,
            tick_id = tick_id
        )
        key_list = [('price', 'LastPrice'), ('volume', 'Volume'), ('openInterest','OpenInterest'),
                    ('open', 'OpenPrice'), ('high', 'HighestPrice'), ('low', 'LowestPrice'),
                    ('prev_close', 'PreClosePrice'), ( 'up_limit', 'UpperLimitPrice'), ('down_limit', 'LowerLimitPrice'),
                    ('bid_price1', 'BidPrice1'), ('ask_price1', 'AskPrice1'),
                    ('bid_vol1', 'BidVolume1'), ('ask_vol1', 'AskVolume1'),
                    #('bidPrice2', 'BidPrice2'), ('askPrice2', 'AskPrice2'),
                    #('bidVol2', 'BidVolume2'), ('askVol2', 'AskVolume2'),
                    #('bidPrice3', 'BidPrice3'), ('askPrice3', 'AskPrice3'),
                    #('bidVol3', 'BidVolume3'), ('askVol3', 'AskVolume3'),
                    #('bidPrice4', 'BidPrice4'), ('askPrice4', 'AskPrice4'),
                    #('bidVol4', 'BidVolume4'), ('askVol4', 'AskVolume4'),
                    #('bidPrice5', 'BidPrice5'), ('askPrice5', 'AskPrice5'),
                    #('bidVol5', 'BidVolume5'), ('askVol5', 'AskVolume5'),
                    ]

        for (tick_key, data_key) in key_list:
            try:
                setattr(tick, tick_key, data[data_key])
            except:
                pass
        product = inst2product(tick.instID)
        if not check_trading_range(tick_id, product, tick.exchange, self.md_data_buffer):
            return 
        else:
            self.on_tick(tick)

    def rsp_qry_account(self, event):
        data = event.dict['data']
        self.qry_acct_data['preBalance'] = data['PreBalance']
        self.qry_acct_data['available'] = data['Available']
        self.qry_acct_data['commission'] = data['Commission']
        self.qry_acct_data['margin'] = data['CurrMargin']
        self.qry_acct_data['closeProfit'] = data['CloseProfit']
        self.qry_acct_data['positionProfit'] = data['PositionProfit']
        self.qry_acct_data['balance'] = (data['PreBalance'] - data['PreCredit'] - data['PreMortgage'] +
                           data['Mortgage'] - data['Withdraw'] + data['Deposit'] +
                           data['CloseProfit'] + data['PositionProfit'] + data['CashIn'] -
                           data['Commission'])

    def rsp_qry_instrument(self, event):
        data = event.dict['data']
        last = event.dict['last']
        if data['ProductClass'] in [THOST_FTDC_PC_Futures, THOST_FTDC_PC_Options] and data['ExchangeID'] in ['CZCE', 'DCE', 'SHFE', 'CFFEX', 'INE',]:
            cont = {}
            cont['instID'] = data['InstrumentID']           
            margin_l = data['LongMarginRatio']
            if margin_l >= 1.0:
                margin_l = 0.0
            cont['margin_l'] = margin_l
            margin_s = data['ShortMarginRatio']
            if margin_s >= 1.0:
                margin_s = 0.0
            cont['margin_s'] = margin_s
            cont['start_date'] =data['OpenDate']
            cont['expiry'] = data['ExpireDate']
            cont['product_code'] = data['ProductID']
            #cont['exchange'] = data['ExchangeID']
            instID = cont['instID']
            self.qry_inst_data[instID] = cont
        if last and self.auto_db_update:
            print("update contract table, new inst # = %s" % len(self.qry_inst_data))
            for instID in self.qry_inst_data:
                expiry = self.qry_inst_data[instID]['expiry']
                try:
                    #expiry_date = datetime.datetime.strptime(expiry, '%Y%m%d')
                    insert_cont_data(self.qry_inst_data[instID])
                except:
                    print(instID, expiry)
                    continue

    def rsp_qry_investor(self, event):
        pass

    def rsp_qry_position(self, event):
        pposition = event.dict['data']
        isLast = event.dict['last']
        if 'InstrumentID' in pposition:
            instID = pposition['InstrumentID']
            if len(instID) ==0:
                return
        else:
            print("no existing position is found for %s" % self.gateway_name)
            return
        if (instID not in self.qry_pos):
            self.qry_pos[instID]   = {'tday': [0, 0], 'yday': [0, 0]}
        key = 'yday'
        idx = 1
        if pposition['PosiDirection'] == '2':
            if pposition['PositionDate'] == '1':
                key = 'tday'
                idx = 0
            else:
                idx = 0
        else:
            if pposition['PositionDate'] == '1':
                key = 'tday'
        self.qry_pos[instID][key][idx] = pposition['Position']
        self.qry_pos[instID]['yday'][idx] = pposition['YdPosition']
        if isLast:
            print(self.qry_pos)

    def rsp_qry_order(self, event):
        sorder = event.dict['data']
        isLast = event.dict['last']
        try:
            order_ref = sorder['OrderRef']
        except:
            return
        if not sorder['OrderRef'].isdigit():
            return
        local_id = order_ref.strip()
        if (local_id in self.id2order):
            iorder = self.id2order[local_id]
            self.system_orders.append(local_id)
            if iorder.status not in [OrderStatus.Cancelled, OrderStatus.Done]:
                status = iorder.on_order(sys_id = sorder['OrderSysID'].strip(), price = sorder['LimitPrice'], volume = sorder['VolumeTraded'])
                if status:
                    event = Event(type=EVENT_ETRADEUPDATE)
                    event.dict['trade_ref'] = iorder.trade_ref
                    self.event_engine.put(event)
                elif sorder['OrderStatus'] in [THOST_FTDC_OST_NoTradeQueueing, THOST_FTDC_OST_PartTradedQueueing, THOST_FTDC_OST_Unknown]:
                    if iorder.status != OrderStatus.Sent:
                        iorder.status = OrderStatus.Sent
                        logContent = 'order status for OrderSysID = %s, Inst=%s is set to %s, but should be waiting in exchange queue' % (iorder.sys_id, iorder.instrument, iorder.status)
                        self.on_log(logContent, level = logging.INFO)
                elif sorder['OrderStatus'] in [THOST_FTDC_OST_Canceled, THOST_FTDC_OST_PartTradedNotQueueing, THOST_FTDC_OST_NoTradeNotQueueing]:
                    if iorder.status != OrderStatus.Cancelled:
                        iorder.on_cancel()
                        event = Event(type=EVENT_ETRADEUPDATE)
                        event.dict['trade_ref'] = iorder.trade_ref
                        self.event_engine.put(event)
                        logContent = 'order status for OrderSysID = %s, Inst=%s is set to %s, but should be cancelled' % (iorder.sys_id, iorder.instrument, iorder.status)
                        self.on_log(logContent, level = logging.INFO)
        if isLast:
            if self.cancel_by_qry_order:
                for local_id in self.id2order:
                    if (local_id not in self.system_orders):
                        iorder = self.id2order[local_id]
                        if iorder.status in Alive_Order_Status:
                            iorder.on_cancel()
                            event = Event(type=EVENT_ETRADEUPDATE)
                            event.dict['trade_ref'] = iorder.trade_ref
                            self.event_engine.put(event)
                            logContent = 'order_ref=%s (Inst=%s,status=%s)is canncelled by qryOrder' % (local_id, iorder.instrument, iorder.status)
                            self.on_log(logContent, level=logging.WARNING)
            self.system_orders = []

    def err_order_insert(self, event):
        '''
            ctp/交易所下单错误回报，不区分ctp和交易所正常情况下不应当出现s
        '''
        porder = event.dict['data']
        error = event.dict['error']
        if not porder['OrderRef'].isdigit():
            return
        local_id = porder['OrderRef'].strip()
        inst = porder['InstrumentID']
        if local_id in self.id2order:
            myorder = self.id2order[local_id]
            inst = myorder.instrument
            myorder.on_cancel()
            event = Event(type=EVENT_ETRADEUPDATE)
            event.dict['trade_ref'] = myorder.trade_ref
            self.event_engine.put(event)
        logContent = 'OrderInsert is not accepted by CTP, local_id=%s, instrument=%s. ' % (local_id, inst)
        if inst not in self.order_stats:
            self.order_stats[inst] = {'submit': 0, 'cancel':0, 'failure': 0, 'status': True }
        self.order_stats[inst]['failure'] += 1
        if (self.order_stats[inst]['failure'] >= self.order_constraints['failure_limit']) and self.order_stats[inst]['status']:
            self.order_stats[inst]['status'] = False
            logContent += 'Failed order reaches the limit, disable instrument = %s' % inst
        self.on_log(logContent, level = logging.WARNING)

    def err_order_action(self, event):
        '''
            ctp/交易所撤单错误回报，不区分ctp和交易所必须处理，如果已成交，撤单后必然到达这个位置
        '''
        porder = event.dict['data']
        error = event.dict['error']
        inst = porder['InstrumentID']        
        if porder['OrderRef'].isdigit():
            local_id = porder['OrderRef'].strip()
            myorder = self.id2order[local_id]
            inst = myorder.instrument
            if (int(error['ErrorID']) in [25, 26]) and myorder.status not in [OrderStatus.Cancelled, OrderStatus.Done]:
                #myorder.on_cancel()
                #event = Event(type=EVENT_ETRADEUPDATE)
                #event.dict['trade_ref'] = myorder.trade_ref
                #self.event_engine.put(event)
                logContent = 'Order Cancel is wrong, local_id=%s, instrument=%s. ' % (porder['OrderRef'], inst)
                self.on_log(logContent, level = logging.WARNING)
                self.qry_commands.append(self.tdApi.qryOrder)
        else:
            self.qry_commands.append(self.tdApi.qryOrder)
        if inst not in self.order_stats:
            self.order_stats[inst] = {'submit': 0, 'cancel':0, 'failure': 0, 'status': True }
        self.order_stats[inst]['failure'] += 1
        if (self.order_stats[inst]['failure'] >= self.order_constraints['failure_limit']) and self.order_stats[inst]['status']:
            self.order_stats[inst]['status'] = False
            logContent = 'Failed order reaches the limit, disable instrument = %s' % inst
            self.on_log(logContent, level = logging.WARNING)


if __name__ == '__main__':
    pass
