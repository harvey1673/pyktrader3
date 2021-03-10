# -*- coding: utf-8 -*-
from .ctp_gateway import *

class CtpSimGateway(CtpGateway):
    def __init__(self, agent, gateway_name='CTP'):
        """Constructor"""
        super(CtpSimGateway, self).__init__(agent, gateway_name, md_api = "ctp.vnctp_gateway.VnctpMdApi", td_api = "ctp.ctpsim_gateway.SimctpTdApi")
        self.qry_enabled = False         # 是否要启动循环查询
        self.md_data_buffer = 0

    def connect(self):
        fileName = self.file_prefix + 'connect.json'
        with open(fileName, 'r') as f:
            setting = json.load(f)
        try:
            userID = str(setting['userID'])
            password = str(setting['password'])
            brokerID = str(setting['brokerID'])
            mdAddress = str(setting['mdAddress'])
        except KeyError:
            logContent = '连接配置缺少字段，请检查'
            self.on_log(logContent, level = logging.WARNING)
            return

        # 创建行情和交易接口对象
        self.mdApi.connect(userID, password, brokerID, mdAddress)
        self.tdApi.connect()

    def register_event_handler(self):
        self.event_engine.register(EVENT_MARKETDATA+self.gateway_name, self.rsp_market_data)
        self.event_engine.register(EVENT_ERRORDERCANCEL+self.gateway_name, self.err_order_insert)
        self.event_engine.register(EVENT_ERRORDERINSERT+self.gateway_name, self.err_order_action)
        self.event_engine.register(EVENT_RTNTRADE+self.gateway_name, self.rtn_trade)
        #self.event_engine.register(EVENT_RTNORDER+self.gateway_name, self.rtn_order)

    def rsp_td_login(self, event):
        pass

        
class SimctpTdApi(object):
    def __init__(self, gateway):
        """API对象的初始化函数"""
        self.gateway = gateway                  # gateway对象
        self.gateway_name = gateway.gateway_name  # gateway对象名称

        self.reqID = 0

        self.connect_status = True      # 连接状态
        self.login_status = True            # 登录状态

        self.userID = ''
        self.password = ''
        self.brokerID = ''
        self.address = ''

        self.frontID = 0
        self.sessionID = 0

    def sendOrder(self, iorder):
        """发单"""
        iorder.local_id = str(iorder.order_ref)
        self.reqID += 1
        req = {}
        req['InstrumentID'] = iorder.instrument
        req['LimitPrice'] = iorder.limit_price
        req['VolumeTotalOriginal'] = iorder.volume
        req['Direction'] = DIRECTION_CMQ2CTP.get(iorder.direction, "")
        req['CombOffsetFlag'] = OFFSET_CMQ2CTP.get(iorder.action_type, "")
        req['OrderPriceType'] = ORDERTYPE_CMQ2CTP.get(iorder.price_type, "")
        req['OrderRef'] = iorder.local_id
        req['InvestorID'] = self.userID
        req['UserID'] = self.userID
        req['BrokerID'] = self.brokerID
        req['CombHedgeFlag'] = THOST_FTDC_HF_Speculation                    # 投机单
        req['ContingentCondition'] = THOST_FTDC_CC_Immediately              # 立即发单
        req['ForceCloseReason'] = THOST_FTDC_FCC_NotForceClose              # 非强平
        req['IsAutoSuspend'] = 0                                            # 非自动挂起
        req['TimeCondition'] = THOST_FTDC_TC_GFD                            # 今日有效
        req['VolumeCondition'] = THOST_FTDC_VC_AV                           # 任意成交量
        req['MinVolume'] = 1                                                # 最小成交量为1
        self.reqOrderInsert(req, self.reqID)

    def reqOrderInsert(self, order, request_id):
        oid = order['OrderRef']
        trade= {'InstrumentID' : order['InstrumentID'],
                'Direction': order['Direction'],
                'Price': order['LimitPrice'],
                'Volume': order['VolumeTotalOriginal'],
                'OrderRef': oid,
                'TradeID': oid,
                'OrderSysID': oid,
                'BrokerOrderSeq': int(oid),
                'OrderLocalID': oid,
                'TradeTime': time.strftime('%H%M%S')}
        event1 = Event(type=EVENT_RTNTRADE+self.gateway_name)
        event1.dict['data'] = trade
        self.gateway.event_engine.put(event1)

    def cancelOrder(self, iorder):
        """撤单"""
        self.reqID += 1
        req = {}
        req['InstrumentID'] = iorder.instrument
        req['ExchangeID'] = iorder.exchange
        req['ActionFlag'] = THOST_FTDC_AF_Delete
        req['BrokerID'] = self.brokerID
        req['InvestorID'] = self.userID
        req['OrderSysID'] = str(iorder.order_ref)
        req['OrderRef'] = str(iorder.order_ref)
        req['FrontID'] = self.frontID
        req['SessionID'] = self.sessionID
        self.reqOrderAction(req, self.reqID)

    def reqOrderAction(self, corder, request_id):
        local_id = corder['OrderRef'].strip()
        if (local_id in self.gateway.id2order):
            myorder = self.gateway.id2order[local_id]
            myorder.on_cancel()
            event = Event(type=EVENT_ETRADEUPDATE)
            event.dict['trade_ref'] = myorder.trade_ref
            self.gateway.event_engine.put(event)

    def reqQryTradingAccount(self,req,req_id=0):
        pass

    def reqQryInstrument(self,req,req_id=0):
        pass

    def reqQryInstrumentMarginRate(self,req,req_id=0):
        pass

    def reqQryInvestorPosition(self,req,req_id=0):
        pass

    def qry_position(self):
        pass

    def qry_account(self):
        pass

    def qry_trade(self):
        pass

    def qry_order(self):
        pass

    def qry_instrument(self):
        pass

    def close(self):
        pass

    def connect(self):
        pass
